import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashMap;

public class DataLoader {

	private String dataFile;
	
	public DataLoader(String dataFile) {
		this.dataFile = dataFile;
	}
	
	public LinkedHashMap<Integer, int[]> load(int part, int numParts) {
		LinkedHashMap<Integer, String> docIDMap = new LinkedHashMap<Integer, String>();
		LinkedHashMap<Integer, int[]> docIDWordMap = new LinkedHashMap<Integer, int[]>();
		
		try {
			
			FileReader fr = new FileReader(this.dataFile);
			BufferedReader br = new BufferedReader(fr);
			int i = 1;
			while (true) {
				String line = br.readLine();
				if (line == null) {
					break;
				} else if ((i-1) % numParts == part) {
					docIDMap.put(i,line);
				}
				i++;
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		for (Integer key : docIDMap.keySet()) {
			String[] tokens = docIDMap.get(key).split(",");
			int[] w = new int[tokens.length];
			for (int j = 0; j < tokens.length; j++) {
				w[j] = Integer.parseInt(tokens[j]);
			}
			docIDWordMap.put(key, w);
		}
		return docIDWordMap;
	}

}
