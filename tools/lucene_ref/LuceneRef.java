// Reference oracle for the HybridSearch lexical engine: runs Lucene itself.
//
// The analyzer chain below is copied from Anserini's
// io.anserini.analysis.DefaultEnglishAnalyzer.createComponents (newDefaultInstance():
// Porter stemmer, EnglishAnalyzer.ENGLISH_STOP_WORDS_SET, no stem exclusions):
//   StandardTokenizer -> EnglishPossessiveFilter -> LowerCaseFilter -> StopFilter -> PorterStemFilter
// Query construction copies Anserini's BagOfWordsQueryGenerator (one SHOULD TermQuery per
// distinct analyzed term, boosted by its count), and ranking copies Anserini's
// BREAK_SCORE_TIES_BY_DOCID sort (score desc, then docid as a STRING ascending).
//
// Build/run with scripts in tools/lucene_ref/run.sh. Modes:
//   analyze   stdin "id\ttext" lines -> stdout "id\tanalyzed tokens\traw tokenizer tokens"
//             (tokens separated by U+0001 so any token text survives)
//   smallfloat  prints "i\tintToByte4(i)" for a set of ints and "b\tbyte4ToInt(b)" for all bytes
//   score  <collection.tsv> <queries.tsv> <k> <k1> <b>
//          builds an in-memory Lucene index with BM25Similarity(k1,b), searches every query,
//          prints "qid\tdocid\trank\tscore\tfloatbits(hex)"
//   stats  <collection.tsv>  prints docCount, sumTotalTermFreq, avgdl bits, per-doc norm bytes

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.apache.lucene.analysis.*;
import org.apache.lucene.analysis.en.*;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.SmallFloat;

public class LuceneRef {
  static final class AnseriniDefaultEnglish extends StopwordAnalyzerBase {
    AnseriniDefaultEnglish() { super(EnglishAnalyzer.ENGLISH_STOP_WORDS_SET); }
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer source = new StandardTokenizer();
      TokenStream result = source;
      result = new EnglishPossessiveFilter(result);
      result = new LowerCaseFilter(result);
      result = new StopFilter(result, this.stopwords);
      result = new PorterStemFilter(result);
      return new TokenStreamComponents(source, result);
    }
  }

  static final class RawStandard extends Analyzer {
    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
      Tokenizer source = new StandardTokenizer();
      return new TokenStreamComponents(source, source);
    }
  }

  static List<String> analyze(Analyzer a, String text) throws IOException {
    List<String> out = new ArrayList<>();
    try (TokenStream ts = a.tokenStream("contents", text)) {
      CharTermAttribute t = ts.addAttribute(CharTermAttribute.class);
      ts.reset();
      while (ts.incrementToken()) out.add(t.toString());
      ts.end();
    }
    return out;
  }

  static BufferedReader stdin() {
    return new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8), 1 << 20);
  }

  static PrintStream stdout() {
    return new PrintStream(new BufferedOutputStream(new FileOutputStream(FileDescriptor.out), 1 << 20), false, StandardCharsets.UTF_8);
  }

  public static void main(String[] args) throws Exception {
    switch (args[0]) {
      case "analyze" -> {
        Analyzer a = new AnseriniDefaultEnglish();
        Analyzer raw = new RawStandard();
        PrintStream out = stdout();
        BufferedReader in = stdin();
        String line;
        while ((line = in.readLine()) != null) {
          int tab = line.indexOf('\t');
          String id = tab < 0 ? "" : line.substring(0, tab);
          String text = tab < 0 ? line : line.substring(tab + 1);
          out.print(id);
          out.print('\t');
          out.print(String.join("\u0001", analyze(a, text)));
          out.print('\t');
          out.print(String.join("\u0001", analyze(raw, text)));
          out.print('\n');
        }
        out.flush();
      }
      case "smallfloat" -> {
        PrintStream out = stdout();
        for (int i = 0; i < 5000; i++) out.println("i\t" + i + "\t" + (SmallFloat.intToByte4(i) & 0xFF));
        for (long i = 5000; i < Integer.MAX_VALUE; i += 7919) out.println("i\t" + i + "\t" + (SmallFloat.intToByte4((int) i) & 0xFF));
        for (int b = 0; b < 256; b++) out.println("b\t" + b + "\t" + SmallFloat.byte4ToInt((byte) b));
        out.flush();
      }
      case "score" -> score(args);
      case "indexstats" -> indexStats(args);
      case "bench" -> bench(args);
      case "version" -> System.out.println(org.apache.lucene.util.Version.LATEST);
      default -> throw new IllegalArgumentException("unknown mode " + args[0]);
    }
  }

  // indexstats <lucene index dir> <out prefix>: field "contents" statistics of an existing
  // (e.g. Anserini prebuilt) index. Writes <prefix>.stats (docCount, sumTotalTermFreq, terms),
  // <prefix>.terms.tsv ("term\tdf\tttf", index order = bytewise sorted), and <prefix>.norms.bin
  // (one norm byte per passage id, 0xFF.. for missing ids is impossible: bytes are raw, ids dense).
  static void indexStats(String[] args) throws Exception {
    try (DirectoryReader reader = DirectoryReader.open(org.apache.lucene.store.FSDirectory.open(java.nio.file.Paths.get(args[1])))) {
      String prefix = args[2];
      long numTerms = 0;
      try (PrintStream out = new PrintStream(new BufferedOutputStream(new FileOutputStream(prefix + ".terms.tsv"), 1 << 22), false, StandardCharsets.UTF_8)) {
        for (LeafReaderContext ctx : reader.leaves()) {
          if (reader.leaves().size() != 1) throw new IllegalStateException("expected one segment");
          Terms terms = ctx.reader().terms("contents");
          TermsEnum te = terms.iterator();
          BytesRef t;
          while ((t = te.next()) != null) {
            out.print(t.utf8ToString());
            out.print('\t');
            out.print(te.docFreq());
            out.print('\t');
            out.print(te.totalTermFreq());
            out.print('\n');
            numTerms++;
          }
        }
      }
      CollectionStatistics cs = new IndexSearcher(reader).collectionStatistics("contents");
      try (PrintStream out = new PrintStream(new FileOutputStream(prefix + ".stats"), false, StandardCharsets.UTF_8)) {
        out.println("maxDoc\t" + reader.maxDoc());
        out.println("docCount\t" + cs.docCount());
        out.println("sumTotalTermFreq\t" + cs.sumTotalTermFreq());
        out.println("sumDocFreq\t" + cs.sumDocFreq());
        out.println("numTerms\t" + numTerms);
        out.println("luceneVersion\t" + org.apache.lucene.util.Version.LATEST);
      }
      // Norms by passage id (ids are the integers 0..maxDoc-1 in MS MARCO passage).
      byte[] norms = new byte[reader.maxDoc()];
      for (LeafReaderContext ctx : reader.leaves()) {
        NumericDocValues nv = ctx.reader().getNormValues("contents");
        StoredFields sf = ctx.reader().storedFields();
        for (int d = 0; d < ctx.reader().maxDoc(); d++) {
          int pid = Integer.parseInt(sf.document(d, java.util.Set.of("id")).get("id"));
          byte n = 0;
          if (nv != null && nv.advanceExact(d)) n = (byte) nv.longValue();
          norms[pid] = n;
        }
      }
      try (FileOutputStream out = new FileOutputStream(prefix + ".norms.bin")) { out.write(norms); }
    }
  }

  // bench <lucene index dir> <queries.tsv> <k> <limit> <warmup passes>
  // Single-threaded latency of IndexSearcher.search(query, k) (Lucene's default top-k path, which
  // uses its own dynamic pruning) with BM25(0.9, 0.4) and Anserini's bag-of-words query. Query
  // analysis is done up front and excluded, as in hs_lex_bench. Prints exact percentiles.
  static void bench(String[] args) throws Exception {
    int k = Integer.parseInt(args[3]), limit = Integer.parseInt(args[4]), warm = Integer.parseInt(args[5]);
    Analyzer analyzer = new AnseriniDefaultEnglish();
    List<Query> qs = new ArrayList<>();
    try (BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(args[2]), StandardCharsets.UTF_8))) {
      String line;
      while ((line = r.readLine()) != null && qs.size() < limit) {
        List<String> tokens = analyze(analyzer, line.substring(line.indexOf('\t') + 1));
        Map<String, Long> collect = tokens.stream().collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        for (String t : collect.keySet())
          builder.add(new BoostQuery(new TermQuery(new Term("contents", t)), (float) collect.get(t)), BooleanClause.Occur.SHOULD);
        qs.add(builder.build());
      }
    }
    try (DirectoryReader reader = DirectoryReader.open(org.apache.lucene.store.FSDirectory.open(java.nio.file.Paths.get(args[1])))) {
      IndexSearcher searcher = new IndexSearcher(reader);  // no executor: single-threaded
      searcher.setSimilarity(new BM25Similarity(0.9f, 0.4f));
      searcher.setQueryCache(null);
      for (int w = 0; w < warm; w++) for (Query q : qs) searcher.search(q, k);
      double[] lat = new double[qs.size()];
      for (int i = 0; i < qs.size(); i++) {
        long t0 = System.nanoTime();
        searcher.search(qs.get(i), k);
        lat[i] = (System.nanoTime() - t0) / 1e3;
      }
      double mean = Arrays.stream(lat).average().orElse(0);
      double[] s = lat.clone();
      Arrays.sort(s);
      java.util.function.DoubleUnaryOperator pct = p -> s[Math.max(0, Math.min(s.length - 1, (int) Math.ceil(p / 100.0 * s.length) - 1))];
      System.out.printf("{\"engine\": \"lucene %s IndexSearcher.search(q, %d)\", \"n\": %d, \"warmup_passes\": %d, \"mean_us\": %.1f, \"p50_us\": %.1f, \"p90_us\": %.1f, \"p95_us\": %.1f, \"p99_us\": %.1f, \"max_us\": %.1f}%n",
          org.apache.lucene.util.Version.LATEST, k, s.length, warm, mean, pct.applyAsDouble(50), pct.applyAsDouble(90), pct.applyAsDouble(95), pct.applyAsDouble(99), s[s.length - 1]);
    }
  }

  static void score(String[] args) throws Exception {
    String collection = args[1], queries = args[2];
    int k = Integer.parseInt(args[3]);
    float k1 = Float.parseFloat(args[4]), b = Float.parseFloat(args[5]);
    Analyzer analyzer = new AnseriniDefaultEnglish();
    ByteBuffersDirectory dir = new ByteBuffersDirectory();
    IndexWriterConfig cfg = new IndexWriterConfig(analyzer);
    cfg.setSimilarity(new BM25Similarity(k1, b));
    try (IndexWriter w = new IndexWriter(dir, cfg);
         BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(collection), StandardCharsets.UTF_8))) {
      String line;
      while ((line = r.readLine()) != null) {
        int tab = line.indexOf('\t');
        Document d = new Document();
        d.add(new StringField("id", line.substring(0, tab), Field.Store.YES));
        d.add(new BinaryDocValuesField("id", new BytesRef(line.substring(0, tab))));
        d.add(new TextField("contents", line.substring(tab + 1), Field.Store.NO));
        w.addDocument(d);
      }
      w.forceMerge(1);
    }
    Sort sort = new Sort(SortField.FIELD_SCORE, new SortField("id", SortField.Type.STRING_VAL));
    PrintStream out = stdout();
    try (DirectoryReader reader = DirectoryReader.open(dir);
         BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(queries), StandardCharsets.UTF_8))) {
      IndexSearcher searcher = new IndexSearcher(reader);
      searcher.setSimilarity(new BM25Similarity(k1, b));
      StoredFields stored = searcher.storedFields();
      String line;
      while ((line = r.readLine()) != null) {
        int tab = line.indexOf('\t');
        String qid = line.substring(0, tab);
        List<String> tokens = analyze(analyzer, line.substring(tab + 1));
        // Anserini BagOfWordsQueryGenerator.buildQuery
        Map<String, Long> collect = tokens.stream().collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        BooleanQuery.Builder builder = new BooleanQuery.Builder();
        for (String t : collect.keySet())
          builder.add(new BoostQuery(new TermQuery(new Term("contents", t)), (float) collect.get(t)), BooleanClause.Occur.SHOULD);
        TopDocs rs = searcher.search(builder.build(), k, sort, true);
        int rank = 1;
        for (ScoreDoc sd : rs.scoreDocs) {
          String docid = stored.document(sd.doc).get("id");
          out.println(qid + "\t" + docid + "\t" + rank++ + "\t" + sd.score + "\t" + Integer.toHexString(Float.floatToRawIntBits(sd.score)));
        }
      }
    }
    out.flush();
  }
}
