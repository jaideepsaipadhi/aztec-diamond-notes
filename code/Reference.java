import de.jtem.riemann.schottky.Schottky;
import de.jtem.riemann.schottky.SchottkyData;
import de.jtem.blas.ComplexMatrix;
import de.jtem.mfc.field.Complex;

/**
 * Reference period matrices from the jtem riemann library, for regression
 * testing our Python Schottky implementation.
 *
 * Input (stdin), one case per line:
 *     g  Are Aim Bre Bim mu   [Are Aim Bre Bim mu ...]
 * Output, one line per case:
 *     OK  g  B11re B11im B12re B12im ...        (row-major, g x g)
 *     FAIL <message>
 *
 * NOTE their convention: getPeriodMatrix() returns log(mu) on the diagonal for
 * genus 1, i.e. 2*pi*i times the normalized period matrix B used in our code.
 */
public class Reference {
  public static void main(String[] args) throws Exception {
    java.io.BufferedReader in = new java.io.BufferedReader(
        new java.io.InputStreamReader(System.in));
    String line;
    while ((line = in.readLine()) != null) {
      line = line.trim();
      if (line.isEmpty() || line.startsWith("#")) continue;
      String[] t = line.split("\\s+");
      int g = Integer.parseInt(t[0]);
      Complex[] A = new Complex[g], B = new Complex[g], mu = new Complex[g];
      for (int i = 0; i < g; i++) {
        int o = 1 + 5 * i;
        A[i]  = new Complex(Double.parseDouble(t[o]),   Double.parseDouble(t[o+1]));
        B[i]  = new Complex(Double.parseDouble(t[o+2]), Double.parseDouble(t[o+3]));
        mu[i] = new Complex(Double.parseDouble(t[o+4]), 0.0);
      }
      try {
        SchottkyData d = new SchottkyData(g);
        d.set(A, B, mu);
        Schottky s = new Schottky(d, 1e-12);
        ComplexMatrix P = s.getPeriodMatrix();
        StringBuilder sb = new StringBuilder("OK " + g);
        for (int i = 0; i < g; i++)
          for (int j = 0; j < g; j++)
            sb.append(" ").append(P.get(i, j).re).append(" ").append(P.get(i, j).im);
        System.out.println(sb);
      } catch (Throwable e) {
        System.out.println("FAIL " + e.getMessage());
      }
      System.out.flush();
    }
  }
}
