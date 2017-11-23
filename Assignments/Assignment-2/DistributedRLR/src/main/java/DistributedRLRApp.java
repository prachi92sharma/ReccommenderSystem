import org.petuum.jbosen.PsApplication;
import org.petuum.jbosen.PsTableGroup;
import org.petuum.jbosen.table.DoubleTable;
import org.petuum.jbosen.row.double_.DenseDoubleRowUpdate;
import org.petuum.jbosen.row.double_.DoubleRow;
import org.petuum.jbosen.row.double_.DoubleRowUpdate;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.Random;
import org.apache.commons.math3.special.*;

public class DistributedRLRApp extends PsApplication {

	private static final int WEIGHT_VECTOR_TABLE = 0;
	private static final int OVERFLOW = 30;
	private static final String TRAINFILE = "/home/abhi/new_full_train.txt";
	private static final String TESTFILE = "/home/abhi/new_full_test.txt";

	private static String[] targetLabels = { "American_film_directors", "Articles_containing_video_clips",
			"English-language_journals", "Windows_games", "American_people_of_Irish_descent",
			"Deaths_from_myocardial_infarction", "Guggenheim_Fellows", "Columbia_University_alumni",
			"Fellows_of_the_Royal_Society", "Major_League_Baseball_pitchers", "Harvard_University_alumni",
			"American_male_film_actors", "English-language_television_programming", "American_film_actresses",
			"American_male_television_actors", "American_films", "English-language_films", "Black-and-white_films",
			"American_drama_films", "Yale_University_alumni", "English-language_albums",
			"American_television_actresses", "American_comedy_films", "The_Football_League_players",
			"English_footballers", "British_films", "American_military_personnel_of_World_War_II",
			"Association_football_goalkeepers", "Serie_A_players", "Italian_footballers",
			"Association_football_midfielders", "Association_football_forwards", "English_cricketers",
			"Scottish_footballers", "French_films", "Insects_of_Europe", "Italian_films", "German_footballers",
			"Indian_films", "Main_Belt_asteroids", "Asteroids_named_for_people", "Rivers_of_Romania",
			"Russian_footballers", "Villages_in_the_Czech_Republic", "Association_football_defenders",
			"Australian_rules_footballers_from_Victoria_(Australia)", "Hindi-language_films", "Brazilian_footballers",
			"Villages_in_Turkey" };

	private String outputDir;
	private int vocSize;
	private double learningRate;
	private double regularization;
	private int epochs;
	private int staleness;

	public DistributedRLRApp(String outputDir, int vocSize, double learningRate, double regularization, int epochs,
			int staleness) {
		this.outputDir = outputDir;
		this.vocSize = vocSize;
		this.learningRate = learningRate;
		this.regularization = regularization;
		this.epochs = epochs;
		this.staleness = staleness;
	}

	private int wordToID(String word) {
		int id = word.hashCode() % this.vocSize;
		// System.out.println(this.vocSize);
		// System.out.println(word);
		if (id < 0) {
			id = id + this.vocSize;
		}
		return id;
	}

	private Map<Integer, Integer> getWordCounts(String curDoc) {

		String[] words = curDoc.split("\\s+");
		Map<Integer, Integer> wordCounts = new HashMap<>();

		for (int i = 0; i < words.length; i++) {
			int wordID = wordToID(words[i]);
			if (wordCounts.containsKey(wordID)) {
				wordCounts.put(wordID, wordCounts.get(wordID) + 1);
			} else {
				wordCounts.put(wordID, 1);
			}
		}
		return wordCounts;
	}

	private int[] getTrueLabels(String labelString, String[] targetLabels) {
		int[] labels = new int[targetLabels.length];
		for (int i = 0; i < targetLabels.length; i++) {
			labels[i] = labelString.contains(targetLabels[i]) ? 1 : 0;
		}
		return labels;
	}

	private double sigmoid(double score) {
		if (Math.abs(score) > OVERFLOW) {
			score = Math.signum(score) * OVERFLOW;
		}
		return 1 / (1 + Math.exp(-score));
	}

	private double[] getDocPredictions(DoubleTable weightVectors, Map<Integer, Integer> wordCounts) {
		int numOfLabels = targetLabels.length;
		DoubleRow bias = weightVectors.get(this.vocSize);
		double[] score = new double[numOfLabels];

		for (int i = 0; i < numOfLabels; i++) {
			score[i] = (double) bias.get(i);
		}

		for (Integer wordID : wordCounts.keySet()) {
			int count = wordCounts.get(wordID);

			DoubleRow w = weightVectors.get(wordID);
			for (int i = 0; i < numOfLabels; i++) {
				score[i] += (double) w.get(i) * count;
			}
		}

		for (int i = 0; i < numOfLabels; i++) {
			score[i] = sigmoid(score[i]);
		}
		return score;
	}

	private void sgdUpdate(DoubleTable weightVectors, int[] labels, Map<Integer, Integer> wordCounts) {
		int numOfLabels = targetLabels.length;

		// predicted probabilities using current model parameters
		double[] prediction = getDocPredictions(weightVectors, wordCounts);

		// update bias term
		DoubleRowUpdate bias_update = new DenseDoubleRowUpdate(numOfLabels);
		for (int i = 0; i < numOfLabels; i++) {
			bias_update.set(i, this.learningRate * (labels[i] - prediction[i]));
		}
		weightVectors.inc(this.vocSize, bias_update);

		for (int wordID = 0; wordID < this.vocSize; wordID++) {
			DoubleRowUpdate w_update = new DenseDoubleRowUpdate(numOfLabels);
			DoubleRow curr_w = weightVectors.get(wordID);
			int count = 0;
			if (wordCounts.containsKey(wordID)) {
				count = wordCounts.get(wordID);
			}

			for (int i = 0; i < numOfLabels; i++) {
				w_update.set(i, -2 * this.learningRate * this.regularization * (double) curr_w.get(i)
						+ this.learningRate * (labels[i] - prediction[i]) * count);
			}
			weightVectors.inc(wordID, w_update);
		}
	}

	@Override
	public void initialize() {
		// Create weight vector table each of which has numWords columns.
		PsTableGroup.createDenseDoubleTable(WEIGHT_VECTOR_TABLE, staleness, targetLabels.length);
	}

	@Override
	public void runWorkerThread(int threadId) {
		int clientId = PsTableGroup.getClientId();
		DoubleRow temp = PsTableGroup.getDoubleTable(WEIGHT_VECTOR_TABLE).get(this.vocSize);
		PsTableGroup.globalBarrier();

		// Get Data
		int part = PsTableGroup.getNumLocalWorkerThreads() * clientId + threadId;
		int numParts = PsTableGroup.getNumTotalWorkerThreads();
		DataLoader dl = new DataLoader();
		ArrayList<String> lines = dl.load(part, numParts);

		// Get Global Tables.
		DoubleTable weightVectorTable = PsTableGroup.getDoubleTable(WEIGHT_VECTOR_TABLE);

		System.out.println("Client " + clientId + " thread " + threadId + " starting RLR sampling...");
		int iter = 0;
		double[] loss = new double[this.epochs];
		for (iter = 1; iter <= this.epochs; iter++) {
			System.out.println(iter);
			long tic = System.currentTimeMillis();
			this.learningRate = this.learningRate * 0.95;
			for (String trainDoc : lines) {
				String label = trainDoc.split("\\s", 2)[0];
				int[] labels = getTrueLabels(label, targetLabels);
				Map<Integer, Integer> wordCounts = getWordCounts(trainDoc.split("\\s", 2)[1]);
				sgdUpdate(weightVectorTable, labels, wordCounts);
				PsTableGroup.clock();
			}
			loss[iter-1] = calculateLoss(lines, weightVectorTable);
			System.out.println("Client " + clientId + " thread " + threadId + " completed iteration " + (iter + 1)
					+ " and took " + (System.currentTimeMillis() - tic) / 1000);
		}
		
		if (this.staleness !=0)
			PsTableGroup.globalBarrier();

		// Get Prediction Accuracy and Loss
		if (threadId == 0 && clientId == 0) {
			System.out.println("Loss for thread 1");
			System.out.println(Arrays.toString(loss));
			try {
				predictLR(TESTFILE);
				predictLR(TRAINFILE);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public void predictLR(String fileName) throws IOException {

		int correctNumber = 0, totalDoc = 0;
		DoubleTable weightVectorTable = PsTableGroup.getDoubleTable(WEIGHT_VECTOR_TABLE);
		BufferedReader testDataIn = new BufferedReader(new FileReader(fileName));
		String testDoc;
		while ((testDoc = testDataIn.readLine()) != null) {
			String label = testDoc.split("\\s", 2)[0];
			Map<Integer, Integer> wordCounts = getWordCounts(testDoc.split("\\s", 2)[1]);
			double[] predictions = getDocPredictions(weightVectorTable, wordCounts);
			int[] labels = getTrueLabels(label, targetLabels);
			int argmax = 0;
			double maxVal = 0;
			for (int i = 0; i < targetLabels.length; i++) {
				if (predictions[i] > maxVal) {
					maxVal = predictions[i];
					argmax = i;
				}
			}
			if (labels[argmax] == 1) {
				correctNumber += 1;
			}
			totalDoc = totalDoc + 1;
		}
		System.out.println(correctNumber);
		System.out.println(totalDoc);
		System.out.println((double) correctNumber / totalDoc);
		testDataIn.close();
	}

	private double calculateLoss(ArrayList<String> lines, DoubleTable weightVectors) {
		double l = 0;
		int count = 0;
		double l2norm = 0;
		for (String trainDoc : lines) {
			count += 1;
			String label = trainDoc.split("\\s", 2)[0];
			int[] labels = getTrueLabels(label, targetLabels);
			Map<Integer, Integer> wordCounts = getWordCounts(trainDoc.split("\\s", 2)[1]);
			double totalLossProb = getTotalLossProb(wordCounts, weightVectors);
			for (int i = 0; i < labels.length; i++) {
				if (labels[i] == 1) {
					l += Math.log(getLossProb(wordCounts, i, weightVectors) / totalLossProb);
				}
			}
		}
		l2norm = calc_l2_norm(weightVectors);
		return (-1 / (double) count) * l + this.regularization * l2norm;
	}

	private double getLossProb(Map<Integer, Integer> wordCounts, int index, DoubleTable weightVectors) {
		double bias = weightVectors.get(this.vocSize).get(index);
		double score = bias;

		for (Integer wordID : wordCounts.keySet()) {
			int count = wordCounts.get(wordID);
			double w = weightVectors.get(wordID,index);
			score += w * count;
		}
		return Math.exp(score);
	}

	private double getTotalLossProb(Map<Integer, Integer> wordCounts, DoubleTable weightVectors) {
		double totalLossProb = 0;
		int numOfLabels = targetLabels.length;
		DoubleRow bias = weightVectors.get(this.vocSize);
		double[] score = new double[numOfLabels];

		for (int i = 0; i < numOfLabels; i++) {
			score[i] = (double)bias.get(i);
		}

		for (Integer wordID : wordCounts.keySet()) {
			int count = wordCounts.get(wordID);
			DoubleRow w = weightVectors.get(wordID);
			for (int i = 0; i < numOfLabels; i++) {
				score[i] += (double)w.get(i) * count;
			}
		}

		for (int i = 0; i < numOfLabels; i++) {
			totalLossProb += Math.exp(score[i]);
		}

		return totalLossProb;
	}

	private double calc_l2_norm(DoubleTable weightVectors) {
		double norms = 0;
		for (int wordId = 0; wordId < this.vocSize; wordId++) {
			for (int label = 0; label < targetLabels.length; label++) {
				norms = norms + Math.pow(weightVectors.get(wordId, label),2);
			}
		}
		return norms;
	}

}
