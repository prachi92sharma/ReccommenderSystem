import org.kohsuke.args4j.Option;
import org.petuum.jbosen.PsApplication;
import org.petuum.jbosen.PsConfig;
import org.petuum.jbosen.PsTableGroup;

public class DistributedRLR {
  
  private static class CmdArgs extends PsConfig {
	  @Option(name = "-outputDir", required = true, usage = "Path to output dir.")
		public String outputDir = "";

		@Option(name = "-vocSize", required = true, usage = "Number of words in the vocabulary.")
		public int vocSize = 0;

		@Option(name = "-learningRate", required = true, usage = "Learning Rate")
		public double learningRate = 0.0;

		@Option(name = "-regularization", required = true, usage = "Regularization")
		public double regularization = 0.0;

		@Option(name = "-epochs", required = true, usage = "epochs")
		public int epochs = 0;

		@Option(name = "-staleness", required = false, usage = "Staleness of word-topic table. Default = 0")
		public int staleness = 0;
  }
  
  public static void main(String[] args) {
    CmdArgs cmd = new CmdArgs();
    cmd.parse(args);
    DistributedRLRApp rlr = new DistributedRLRApp(cmd.outputDir, cmd.vocSize, cmd.learningRate, cmd.regularization,
			cmd.epochs, cmd.staleness);
    rlr.run(cmd);
  }

}
