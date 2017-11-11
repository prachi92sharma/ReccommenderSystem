import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class RLR {
	
	private static double OVERFLOW = 20;
	private static final String TRAINFILE = "C:\\Users\\abhin\\eclipse-workspace\\LR\\src\\new_full_train.txt";
	private static final String TESTFILE = "C:\\Users\\abhin\\eclipse-workspace\\LR\\src\\new_full_test.txt";
	int vocSize = 0;
	double learningRate = 0;
	double regularization = 0;
	int epochs = 0;
	String targetLabels[];
	Map<Integer, double[]> weightVectors = new HashMap<>();
	Map<Integer, int[]> coeffUpdateLags = new HashMap<>();
	double loss[];
	
	/* Initialize parameters */
	public RLR(int vocSize, double learningRate, double regularization, int epochs, String []targetLabels) {
		this.vocSize = vocSize;
		this.learningRate = learningRate;
		this.regularization = regularization;
		this.epochs = epochs;
		this.targetLabels = targetLabels;
		this.weightVectors.put(-1, new double[this.targetLabels.length]);
		loss = new double[this.epochs];
	}
	
	private int wordToID(String word) {
		int id =  word.hashCode() % this.vocSize;
		if (id < 0) {
			id = id + this.vocSize;
		}
		return id;
	}

	private Map<Integer, Integer> getWordCounts(String curDoc) {
		
		String[] words = curDoc.split("\\s+");
		Map<Integer, Integer> wordCounts = new HashMap<>();
		
		for (int i = 0; i < words.length; i++) {
			words[i] = words[i].replaceAll("\\W", "").toLowerCase();
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
	
	private double[] getDocPredictions(Map<Integer, Integer> wordCounts) {
		int numOfLabels = this.targetLabels.length;
		double []bias = this.weightVectors.get(-1);
		double[] score = new double[numOfLabels];
		
		for (int i = 0; i < numOfLabels; i++) {
			score[i] = bias[i];
		}
		
		for (Integer wordID : wordCounts.keySet()) {
			int count = wordCounts.get(wordID);
			
			if (this.weightVectors.containsKey(wordID)) {
				double[] w = this.weightVectors.get(wordID);
				for (int i = 0; i < numOfLabels; i++) {
					score[i] += w[i] * count;
				}
			}
		}

		for (int i = 0; i < numOfLabels; i++) {
			score[i] = sigmoid(score[i]);
		}
		return score;
	}
	
	private void sgdUpdate(int[] labels, Map<Integer, Integer> wordCounts, int counter) {
		int numOfLabels = targetLabels.length;

		// predicted probabilities using current model parameters
		double[] prediction = getDocPredictions(wordCounts);
		
		// update bias term
		double []bias = this.weightVectors.get(-1).clone();
		this.weightVectors.put(-1, bias);
		
		for (Integer wordID : wordCounts.keySet()) {
			int count = wordCounts.get(wordID);
			
			double[] w = new double[numOfLabels];
			int[] updateLags = new int[numOfLabels];
			
			// Regularization update
			if (this.weightVectors.containsKey(wordID)) {
				
				w = this.weightVectors.get(wordID).clone();
				if (this.coeffUpdateLags.containsKey(wordID)) {
					updateLags = this.coeffUpdateLags.get(wordID).clone();
				}
				
				for (int i = 0; i < numOfLabels; i++) {
					w[i] *= Math.pow (1 - 2 * this.learningRate * this.regularization, counter - updateLags[i]);
				}
			}
				
			for (int i = 0; i < numOfLabels; i++) {
				w[i] += this.learningRate * (labels[i] - prediction[i]) * count;
				updateLags[i] = counter;
			}
			this.weightVectors.put(wordID, w);
			this.coeffUpdateLags.put(wordID, updateLags);
		}
	}

	private void updateRegularization(int counter) {

		for (Integer wordID : this.weightVectors.keySet()) {
			// skip bias term
			if (wordID < 0) {
				continue;
			}
			
			double[] w = this.weightVectors.get(wordID).clone();
			int[] updateLags = coeffUpdateLags.get(wordID).clone();

			for (int i = 0; i < this.targetLabels.length; i++) {
				w[i] *= Math.pow(1 - 2 * this.learningRate * this.regularization, counter - updateLags[i]);
				updateLags[i] = counter;
			}
			this.weightVectors.put(wordID, w);
			this.coeffUpdateLags.put(wordID, updateLags);
		}
	}

	public void fit() throws IOException{
		
		int counter = 0;

		for (int iter = 1; iter <= this.epochs; iter++) {
			System.out.println(iter);
			this.learningRate = this.learningRate * 1;
			BufferedReader trainDataIn = new BufferedReader(new FileReader(TRAINFILE));
			String trainDoc;
			while ((trainDoc = trainDataIn.readLine()) != null) {
				
				counter += 1;
				String label = trainDoc.split("\\s", 2)[0];
				int[] labels = getTrueLabels(label, targetLabels);
				Map<Integer, Integer> wordCounts = getWordCounts(trainDoc.split("\\s", 2)[1]);
				
				sgdUpdate(labels, wordCounts, counter);
			}
			
			updateRegularization(counter);
			/*
			for (Integer key :this.weightVectors.keySet()) {
				double w[] = this.weightVectors.get(key);
				for (int i = 0;i < w.length; i++) {
					System.out.print(w[i]+ " ");
				}
				System.out.println();
			}
			*/
			trainDataIn.close();
			loss[iter-1] = calculateLoss();
			System.out.println("Loss at iter "+iter+ " is "+ loss[iter-1]);
		}
		BufferedWriter outputWriter = null;
		outputWriter = new BufferedWriter(new FileWriter("C:\\Users\\abhin\\eclipse-workspace\\LR\\src\\loss.txt"));
		outputWriter.write(Arrays.toString(loss)); 
		outputWriter.close();
	}

	public void predictLR(String fileName) throws IOException {
		
		int correctNumber = 0, totalDoc = 0;
		
		BufferedReader testDataIn = new BufferedReader(new FileReader(fileName));
		String testDoc;
		while ((testDoc = testDataIn.readLine()) != null) {
			String label = testDoc.split("\\s", 2)[0];
			Map<Integer, Integer> wordCounts = getWordCounts(testDoc.split("\\s", 2)[1]);
			double[] predictions = getDocPredictions(wordCounts);
			int[] labels = getTrueLabels(label,targetLabels);
			int argmax = 0; 
			double maxVal = 0;
			for (int i = 0; i < this.targetLabels.length; i++) {
				if ( predictions[i] > maxVal) {
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
		System.out.println((double)correctNumber / totalDoc);
		testDataIn.close();
	}

	private double calculateLoss() throws IOException {
		double l=0;
	    int count = 0;
	    double l2norm = 0;
	    BufferedReader br = new BufferedReader(new FileReader(TRAINFILE));
	    String trainDoc;
	    while ((trainDoc = br.readLine()) != null) {
	    	count += 1;
	    	String label = trainDoc.split("\\s", 2)[0];
			int[] labels = getTrueLabels(label, targetLabels);
			Map<Integer, Integer> wordCounts = getWordCounts(trainDoc.split("\\s", 2)[1]);
			double totalLossProb = getTotalLossProb(wordCounts);
			for (int i = 0; i< labels.length; i++) {
				if ( labels[i] == 1) {
					l += Math.log(getLossProb(wordCounts,i) / totalLossProb);
				}
			}
	    }
	    br.close();
	    l2norm = calc_l2_norm();
	    return (-1 / (double)count) * l + this.regularization * l2norm;
	}
	
	private double getLossProb(Map<Integer, Integer> wordCounts, int index) {
		double bias = this.weightVectors.get(-1).clone()[index];
		double score = bias;
		
		for (Integer wordID : wordCounts.keySet()) {
			int count = wordCounts.get(wordID);
			if (this.weightVectors.containsKey(wordID)) {
				double w = this.weightVectors.get(wordID).clone()[index];
				score += w * count;
			}
		}

		return Math.exp(score);
	}
	
	private double getTotalLossProb(Map<Integer, Integer> wordCounts) {
		double totalLossProb = 0;
		int numOfLabels = this.targetLabels.length;
		double []bias = this.weightVectors.get(-1);
		double []score = new double[numOfLabels];
		
		for (int i = 0; i < numOfLabels; i++) {
			score[i] = bias[i];
		}
		
		for (Integer wordID : wordCounts.keySet()) {
			int count = wordCounts.get(wordID);
			
			if (this.weightVectors.containsKey(wordID)) {
				double[] w = this.weightVectors.get(wordID);
				for (int i = 0; i < numOfLabels; i++) {
					score[i] += w[i] * count;
				}
			}
		}

		for (int i = 0; i < numOfLabels; i++) {
			totalLossProb += Math.exp(score[i]);
		}
		
		return totalLossProb;
	}
	
	private double calc_l2_norm()
	{
	    double w[];
	    double norms = 0;
	    for (Map.Entry<Integer, double[]> entry : this.weightVectors.entrySet()) {
	         w = entry.getValue();
	         for(int i=0; i < w.length; i++)
	         {
	             norms = norms + w[i] * w[i];
	         }
	    }
	    return norms;
	}

	static class RunRLR{
		public static void main(String []args) throws IOException {
			String[] targetLabels = { "American_film_directors", "Articles_containing_video_clips", "English-language_journals", "Windows_games", "American_people_of_Irish_descent", "Deaths_from_myocardial_infarction", "Guggenheim_Fellows", "Columbia_University_alumni", "Fellows_of_the_Royal_Society", "Major_League_Baseball_pitchers", "Harvard_University_alumni", "American_male_film_actors", "English-language_television_programming", "American_film_actresses", "American_male_television_actors", "American_films", "English-language_films", "Black-and-white_films", "American_drama_films", "Yale_University_alumni", "English-language_albums", "American_television_actresses", "American_comedy_films", "The_Football_League_players", "English_footballers", "British_films", "American_military_personnel_of_World_War_II", "Association_football_goalkeepers", "Serie_A_players", "Italian_footballers", "Association_football_midfielders", "Association_football_forwards", "English_cricketers", "Scottish_footballers", "French_films", "Insects_of_Europe", "Italian_films", "German_footballers", "Indian_films", "Main_Belt_asteroids", "Asteroids_named_for_people", "Rivers_of_Romania", "Russian_footballers", "Villages_in_the_Czech_Republic", "Association_football_defenders", "Australian_rules_footballers_from_Victoria_(Australia)", "Hindi-language_films", "Brazilian_footballers", "Villages_in_Turkey" };
			RLR ob = new RLR(250000, 0.3, 0.001, 20, targetLabels);
			long tic = System.currentTimeMillis();
			ob.fit();
			System.out.println("Training Time -> "+ ((double)System.currentTimeMillis() - tic) / 1000);
			tic = System.currentTimeMillis();
			ob.predictLR(TESTFILE);
			System.out.println("Test Time -> "+ ((double)System.currentTimeMillis() - tic) / 1000);
			ob.predictLR(TRAINFILE);
			System.out.println(Arrays.toString(ob.loss));
		}
	}
}