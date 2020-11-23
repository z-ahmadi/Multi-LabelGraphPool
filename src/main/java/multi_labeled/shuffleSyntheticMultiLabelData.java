package multi_labeled;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

public class shuffleSyntheticMultiLabelData {

	public static void main(String[] args) throws IOException {
		String path = "/home/havij/Dropbox/Scripts-paper codes/Multi-labelGraphPool/testPYP_0.2_0.8_0.5_n3000_t20_d0.75";
		BufferedReader bf = new BufferedReader(new FileReader(path));		
		PrintWriter pw = new PrintWriter(new File(path+"-shuffled"));
		
		int numInst = 3000, numConcept = 3;

		String[][] insts = new String[numInst][numConcept];
		String line;
		int counter = 0;
		
		while((line = bf.readLine())!= null) {
			int col = counter/numInst;
			insts[counter - col*numInst][col] = line;
			counter ++;
		}
		
		// c1,c2,c3,c2,c1,c3,c1,c3,c2
		for(int i=0; i<1000; i++)
			pw.println(insts[i][0]);
		for(int i=0; i<1000; i++)
				pw.println(insts[i][1]);
		for(int i=0; i<1000; i++)
			pw.println(insts[i][2]);

		for(int i=1000; i<2000; i++)
				pw.println(insts[i][1]);
		for(int i=1000; i<2000; i++)
			pw.println(insts[i][0]);
		for(int i=1000; i<2000; i++)
			pw.println(insts[i][2]);

		for(int i=2000; i<3000; i++)
				pw.println(insts[i][0]);
		for(int i=2000; i<3000; i++)
			pw.println(insts[i][2]);
		for(int i=2000; i<3000; i++)
			pw.println(insts[i][1]);
		
		pw.close();
	}

}
