import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class countAttributes {

	public static void main(String[] args) throws FileNotFoundException, IOException {
		String path = "/home/havij/Dropbox/Backup Codes/data/SophieGenerator/New/testPYP_d0.75";
		Instances inst = new Instances(new FileReader(path+".arff"));
		int windowSize = 100;
		int MaxWindow = (int) Math.ceil((double)inst.numInstances()/windowSize);
		
		int[][][] counts = new int[MaxWindow][inst.numAttributes()][2]; //2 because of 0 & 1 values of all attributes
		
		for(int i=0; i<MaxWindow; i++) {
			System.out.print("batch "+i+"\t");
			for(int a=0; a<inst.numAttributes(); a++) {
				for(int s=0; s<windowSize; s++) {
					int value = (int) inst.instance(i*windowSize+s).value(a);
					if(value == 0)
						counts[i][a][0]++;
					else if(value == 1)
						counts[i][a][1]++;
				}
				System.out.print("a"+a+":"+counts[i][a][1]+" , "); //+":n0="+counts[i][a][0]
			}
			System.out.println();
		}
		
		

	}

}
