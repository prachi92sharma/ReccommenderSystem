import org.petuum.jbosen.PsApplication;
import org.petuum.jbosen.PsTableGroup;
import org.petuum.jbosen.table.IntTable;
import org.petuum.jbosen.row.int_.DenseIntRowUpdate;
import org.petuum.jbosen.row.int_.IntRow;
import org.petuum.jbosen.row.int_.IntRowUpdate;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.LinkedHashMap;
import java.util.Random;
import org.apache.commons.math3.special.*;

public class LdaApp extends PsApplication {

	private static final int TOPIC_TABLE = 0;
	private static final int WORD_TOPIC_TABLE = 1;
	private static final int DOC_TOPIC_TABLE = 2;

	private String outputDir;
	private int numWords;
	private int numTopics;
	private double alpha;
	private double beta;
	private int numIterations;
	private int numClocksPerIteration;
	private int staleness;
	private int numOfDocs;
	private DataLoader dataLoader;
	private Random random;

	public LdaApp(String dataFile, String outputDir, int numWords, int numTopics, double alpha, double beta,
			int numIterations, int numClocksPerIteration, int staleness, int numOfDocs) {
		this.outputDir = outputDir;
		this.numWords = numWords;
		this.numTopics = numTopics;
		this.alpha = alpha;
		this.beta = beta;
		this.numIterations = numIterations;
		this.numClocksPerIteration = numClocksPerIteration;
		this.staleness = staleness;
		this.dataLoader = new DataLoader(dataFile);
		this.numOfDocs = numOfDocs;
		this.random = new Random();
	}

	public double logDirichlet(double[] alpha) {
		double sumLogGamma = 0.0;
		double logSumGamma = 0.0;
		for (double value : alpha) {
			sumLogGamma += Gamma.logGamma(value);
			logSumGamma += value;
		}
		return sumLogGamma - Gamma.logGamma(logSumGamma);
	}

	public double logDirichlet(double alpha, int k) {
		return k * Gamma.logGamma(alpha) - Gamma.logGamma(k * alpha);
	}

	public double[] getRows(IntTable matrix, int columnId) {
		double[] rows = new double[this.numWords];
		for (int i = 0; i < this.numWords; i++) {
			rows[i] = (double) matrix.get(i, columnId);
		}
		return rows;
	}

	public double[] getColumns(int[][] matrix, int rowId) {
		double[] cols = new double[this.numTopics];
		for (int i = 0; i < this.numTopics; i++) {
			cols[i] = (double) matrix[rowId][i];
		}
		return cols;
	}

	public double getLogLikelihood(IntTable wordTopicTable, int[][] docTopicTable) {
		double lik = 0.0;
		for (int k = 0; k < this.numTopics; k++) {
			double[] temp = this.getRows(wordTopicTable, k);
			for (int w = 0; w < this.numWords; w++) {
				temp[w] += this.alpha;
			}
			lik += this.logDirichlet(temp);
			lik -= this.logDirichlet(this.beta, this.numWords);
		}
		for (int d = 0; d < docTopicTable.length; d++) {
			double[] temp = this.getColumns(docTopicTable, d);
			for (int k = 0; k < this.numTopics; k++) {
				temp[k] += this.alpha;
			}
			lik += this.logDirichlet(temp);
			lik -= this.logDirichlet(this.alpha, this.numTopics);
		}
		return lik;
	}

	@Override
	public void initialize() {
		/*
		 * Create global topic count table. This table only has one row, which contains
		 * counts for all topics
		 */
		PsTableGroup.createDenseIntTable(TOPIC_TABLE, staleness, this.numTopics);
		/*
		 * Create global word-topic table. This table contains numWords rows, each of
		 * which has numTopics columns
		 */
		PsTableGroup.createDenseIntTable(WORD_TOPIC_TABLE, staleness, this.numTopics);
		/*
		 * Create global word-topic table. This table contains numDocs rows, each of
		 * which has numTopics columns
		 */
		PsTableGroup.createDenseIntTable(DOC_TOPIC_TABLE, staleness, this.numTopics);

	}

	/*
	 * Return weight of a word w under topic k public double phi(int word, int
	 * topic, IntTable wordTopicCount, IntTable topicTable) { double phi = 0; phi =
	 * (wordTopicCount.get(word, topic) + this.beta) / (topicTable.get(0, topic) +
	 * this.numWords * this.beta); return phi; }
	 */

	public void intializeGibbs(LinkedHashMap<Integer, int[]> w, int[][] docTopicTable, IntTable wordTopicTable,
			IntTable topicTable, int[][] z) {

		IntRow temp = topicTable.get(0);
		int i = 0;
		for (Integer key : w.keySet()) {
			for (int j = 0; j < w.get(key).length; j++) {

				int word = w.get(key)[j];
				int randomTopic = random.nextInt(this.numTopics);
				z[i][j] = randomTopic;
				docTopicTable[i][randomTopic] += 1;

				/* Update Count of Topic for a Word */
				temp = wordTopicTable.get(word);
				IntRowUpdate count_update = new DenseIntRowUpdate(this.numTopics);
				count_update.set(randomTopic, 1);
				wordTopicTable.inc(word, count_update);

				/* Update Total Topic Count */
				count_update = new DenseIntRowUpdate(this.numTopics);
				count_update.set(randomTopic, 1);
				topicTable.inc(0, count_update);
			}
			i += 1;
		}
	}

	public int resample(int word, int docId, int j, IntTable wordTopicTable, IntTable topicTable, int[][] docTopicTable,
			int[][] z) {
		int topic = 0;
		double norm = 0;
		double[] p = new double[this.numTopics];
		for (int k = 0; k < this.numTopics; k++) {
			int isTopicEqualToWordTopic = z[docId][j] == k ? 1 : 0;
			double a_k = docTopicTable[docId][k] - isTopicEqualToWordTopic + this.alpha;
			double b_k = ((double) wordTopicTable.get(word, k) - isTopicEqualToWordTopic + this.beta)
					/ ((double) topicTable.get(0, k) - isTopicEqualToWordTopic + this.numWords * this.beta);
			double p_k = a_k * b_k;
			p[k] = p_k;
			norm += p_k;
		}

		double sumUptoK = 0.0;
		double r = Math.random();
		for (int k = 0; k < this.numTopics; k++) {
			sumUptoK += p[k] / norm;
			if (r < sumUptoK) {
				topic = k;
				break;
			}
		}
		return topic;
	}

	public void flip(int word, int docId, int j, int old_topic, int new_topic, IntTable wordTopicTable,
			IntTable topicTable, int[][] docTopicTable, int[][] z) {

		if (old_topic != new_topic) {
			z[docId][j] = new_topic;

			/* Decrement Operation */
			docTopicTable[docId][old_topic] -= 1;

			IntRowUpdate count_update = new DenseIntRowUpdate(this.numTopics);
			count_update.set(old_topic, -1);
			wordTopicTable.inc(word, count_update);

			count_update = new DenseIntRowUpdate(this.numTopics);
			count_update.set(old_topic, -1);
			topicTable.inc(0, count_update);

			/* Increment Operations */
			docTopicTable[docId][new_topic] += 1;

			count_update = new DenseIntRowUpdate(this.numTopics);
			count_update.set(new_topic, 1);
			wordTopicTable.inc(word, count_update);

			count_update = new DenseIntRowUpdate(this.numTopics);
			count_update.set(new_topic, 1);
			topicTable.inc(0, count_update);
		}
	}

	public void updateGlobalDocumentTopicTable(IntTable documentTopicTable, int[][] docTopicTable, Object[] keyset) {
		for (int i = 0; i < docTopicTable.length; i++) {
			IntRowUpdate update = new DenseIntRowUpdate(this.numTopics);
			for (int k = 0; k < this.numTopics; k++) {
				update.set(k, docTopicTable[i][k]);
			}
			documentTopicTable.inc((int) keyset[i], update);
		}
	}

	@Override
	public void runWorkerThread(int threadId) {
		int clientId = PsTableGroup.getClientId();

		/* Load data for this thread */
		System.out.println("Client " + clientId + " thread " + threadId + " loading data...");
		int part = PsTableGroup.getNumLocalWorkerThreads() * clientId + threadId;
		int numParts = PsTableGroup.getNumTotalWorkerThreads();
		LinkedHashMap<Integer, int[]> w = this.dataLoader.load(part, numParts);
		System.out.println("Client " + clientId + " thread " + threadId + " loaded " + w.size() + " lines");

		/* Get global tables */
		IntTable topicTable = PsTableGroup.getIntTable(TOPIC_TABLE);
		IntTable wordTopicTable = PsTableGroup.getIntTable(WORD_TOPIC_TABLE);
		IntTable documentTopicTable = PsTableGroup.getIntTable(DOC_TOPIC_TABLE);

		/* Initialize LDA variables */
		System.out.println("Client " + clientId + " thread " + threadId + " initializing variables...");
		int[][] docTopicTable = new int[w.size()][this.numTopics];

		/* Stores the last topic seen for each word per document */
		int[][] z = new int[w.size()][];
		int i = 0;
		for (Integer key : w.keySet()) {
			z[i] = new int[w.get(key).length];
			i += 1;
		}

		intializeGibbs(w, docTopicTable, wordTopicTable, topicTable, z);

		/* Global barrier to synchronize word-topic table */
		PsTableGroup.globalBarrier();

		/* Do LDA Gibbs sampling */
		System.out.println("Client " + clientId + " thread " + threadId + " starting gibbs sampling...");
		double[] llh = new double[this.numIterations];
		double[] sec = new double[this.numIterations];
		double totalSec = 0.0;
		for (int iter = 0; iter < this.numIterations; iter++) {
			long startTime = System.currentTimeMillis();
			Object[] keyset = w.keySet().toArray();
			for (int batch = 0; batch < this.numClocksPerIteration; batch++) {

				int begin = w.size() * batch / this.numClocksPerIteration;
				int end = w.size() * (batch + 1) / this.numClocksPerIteration;

				for (int docId = begin; docId < end; docId++) {
					for (int j = 0; j < w.get((int) keyset[docId]).length; j++) {
						int topic = resample(w.get((int) keyset[docId])[j], docId, j, wordTopicTable, topicTable,
								docTopicTable, z);
						flip(w.get((int) keyset[docId])[j], docId, j, z[docId][j], topic, wordTopicTable, topicTable,
								docTopicTable, z);
					}
				}
				PsTableGroup.clock();
			}

			/* Calculate likelihood and elapsed time */
			totalSec += (double) (System.currentTimeMillis() - startTime) / 1000;
			sec[iter] = totalSec;
			llh[iter] = this.getLogLikelihood(wordTopicTable, docTopicTable);
			System.out.println("Client " + clientId + " thread " + threadId + " completed iteration " + (iter + 1)
					+ "\n    Elapsed seconds: " + sec[iter] + "\n    Log-likelihood: " + llh[iter]);
		}

		PsTableGroup.globalBarrier();
		updateGlobalDocumentTopicTable(documentTopicTable, docTopicTable, w.keySet().toArray());
		PsTableGroup.globalBarrier();
		
		if (clientId == 0 && threadId == 0) {
			System.out.println("Client " + clientId + " thread " + threadId + " writing likelihood to file..."); 
			try { 
				PrintWriter out = new PrintWriter(this.outputDir + "/likelihood_" + this.numTopics +".csv"); 
				for (i = 0; i < this.numIterations; i++) { 
					out.println((i + 1) + "," + sec[i] + "," + llh[i]); 
				}
				out.close(); 
			} catch (IOException e) {
			  e.printStackTrace(); 
			  System.exit(1); 
			}
		}
		 
		// Output tables and print top 20 words per topic
		if (clientId == 0 && threadId == 0) {
			System.out.println("Client " + clientId + " thread " + threadId + " writing word-topic table to file...");
			try {
				PrintWriter out = new PrintWriter(this.outputDir + "/word-topic.csv");
				for (i = 0; i < this.numWords; i++) {
					out.print(wordTopicTable.get(i, 0));
					for (int k = 1; k < this.numTopics; k++) {
						out.print("," + wordTopicTable.get(i, k));
					}
					out.println();
				}
				out.close();
			} catch (IOException e) {
				e.printStackTrace();
				System.exit(1);
			}

			System.out.println("Client " + clientId + " thread " + threadId + " writing document-topic table to file...");
			try {
				PrintWriter out = new PrintWriter(this.outputDir + "/doc-topic.csv");
				for (i = 1; i <= this.numOfDocs; i++) {
					int sumOfDoc = 0;
					for (int k = 0; k < this.numTopics; k++) {
						sumOfDoc += documentTopicTable.get(i, k);

					}
					double theta = (documentTopicTable.get(i, 0));
					out.print(theta);
					for (int k = 1; k < this.numTopics; k++) {
						theta = (documentTopicTable.get(i, k));
						out.print("," + theta);
					}
					out.println();
				}
				out.close();
			} catch (IOException e) {
				e.printStackTrace();
				System.exit(1);
			}
			try {
				PrintWriter out1 = new PrintWriter(this.outputDir + "/doc-topic-normalized.csv");
				for (i = 1; i <= this.numOfDocs; i++) {
					int sumOfDoc = 0;
					for (int k = 0; k < this.numTopics; k++) {
						sumOfDoc += documentTopicTable.get(i, k);
					}
					
					double theta = (documentTopicTable.get(i, 0)+ this.alpha) / (sumOfDoc + this.numTopics * this.alpha);
					out1.print(theta);
					for (int k = 1; k < this.numTopics; k++) {
						theta = (documentTopicTable.get(i, k) + this.alpha) / (sumOfDoc + this.numTopics * this.alpha);
						out1.print("," + theta);
					}
					out1.println();
				}
				out1.close();
			} catch (IOException e) {
				e.printStackTrace();
				System.exit(1);
			}
		}

		PsTableGroup.globalBarrier();
		System.out.println("Client " + clientId + " thread " + threadId + " exited.");
	}
}
