package multi_labeled;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.StringTokenizer;
/**
 * it calculates the average of all the measures for any number of runs but only one classifier
 * the files should be in the format of: pathOfMethod/.../measure/runNumber.txt
 * 
 * edit: on 24.4.2017 I added the averaging for offline experiment as well. 
 * Now the first line is the results for online RACE-AE & second line for offline variant
 *
 */

public class AverageOfMeasures {
	static ArrayList<File> allFiles;
	static ArrayList<String> measures = new ArrayList<String>();
	static ArrayList<String> alg = new ArrayList<String>();
	
	public static void main(String[] args) throws IOException {
		//make the averages of all measures (the folder is outputPath)
//		System.out.println(args[0]);
//		if(args[0].endsWith("/"))
//			args[0] = args[0].substring(0, args[0].length()-1);
//		String path = "/home/havij/Dropbox/Backup Codes/results/rcv1v2subset1_500_NaiveBayesUpdateable/7/repeat-RACE";
		String path ="/media/havij/dataDrive/RACE-experimental results/results-lenovo-combinedWithDropbox/newRepeat/enron_100_NaiveBayesUpdateable/6/repeat-RACE";
//		String path = "/home/havij/Dropbox/Backup Codes/results/batchIters/mediamill_500_NaiveBayesUpdateable/7/repeat-RACE";
//		String path = "/home/havij/Dropbox/Backup Codes/results/nus-wide-bow_500_NaiveBayesUpdateable/7/repeat-RACE";
		AverageOfMeasures am = new AverageOfMeasures();
		am.calculateAvgOfMultipleMethods(path, true);
//		am.calculateAvgOfOneMethod(path,true);
	}
	
	
	public void calculateAvgOfMultipleMethods(String args, boolean offline) throws IOException {	
		String datasetPath = args;
//		boolean offlineFlag = false;
		
		allFiles = new ArrayList<File>();
		returnFilePath(datasetPath);
		
		PrintWriter pw;
//		double[] offValues = new double[measures.size()];
		int[] runs = new int[measures.size()];
		int iterNum = 0, batchNum = 0;
		
		//set iter and line values
		BufferedReader bft = new BufferedReader(new FileReader(allFiles.get(0)));
		System.out.println(allFiles.get(0));
		String line;
		int lineNum=0; 
		while((line = bft.readLine())!= null){
			StringTokenizer stg = new StringTokenizer(line, ",");
			boolean emptyLine = true;
			int colNum = 0;
			while(stg.hasMoreTokens()) {
				String val = stg.nextToken().trim();
				if(val.length() > 0){
					colNum++;
					emptyLine = false;
				}
			}
			if(iterNum == 0)
				iterNum = colNum;
			
			if(!emptyLine)
				lineNum++;
		}
		batchNum = lineNum;
		
		System.out.println(batchNum+" "+iterNum);
		

		double[][][] avgValues = new double[measures.size()][batchNum][iterNum]; //[measure][batch][iter]
		
		for(int i=0; i<allFiles.size(); i++){
			String path = allFiles.get(i).getPath();
			if(!path.contains("threshold") && !path.contains("snapshot") && !path.contains("BetaSeq") && !path.contains("ChangeLogs") && !path.contains("detailedMeasures")){
				String f = path.substring(0,path.lastIndexOf("/"));	//folder
				String meas = f.substring(f.lastIndexOf("/")+1, f.length());	//measure
				int ind = measures.indexOf(meas);
				runs[ind]++;
				
				BufferedReader bf = new BufferedReader(new FileReader(allFiles.get(i)));
				System.out.println(allFiles.get(i));
				lineNum = 0;
				if(meas.contains("time")) {
					
				}else {
					while(lineNum < batchNum){
						line = bf.readLine();
//						System.out.println(line);
						StringTokenizer stg = new StringTokenizer(line, ",");
						int colNum = 0;
						while(stg.hasMoreTokens()) {
							String val = stg.nextToken().trim();
							if(val.length() > 0){
								avgValues[ind][lineNum][colNum] += new Double(val);
								colNum++;
							}
						}
						
						lineNum++;
					}
				}
				
			}
		}
		
		//write averages in file
				File f = new File(datasetPath+"-Avg/");
				f.mkdirs();
				
				for(int k=0; k<measures.size(); k++){					
					if(!measures.get(k).contains("thresholds") && !measures.get(k).contains("BetaSeq")) {
						pw = new PrintWriter(new File(datasetPath+"-Avg/"+measures.get(k).replaceAll(" ", "-")+".txt"));
					
						for(int b=0; b<batchNum; b++){ 
							for(int i=0; i<iterNum; i++) {
								pw.print(String.format( "%.5f", avgValues[k][b][i]/runs[k]) + "\t");
							}
							pw.println();
						}
						
						pw.close();
					}
				}
				
		
	}
	
	
	public void calculateAvgOfOneMethod(String args, boolean offline) throws IOException {	
		String datasetPath = args;
//		boolean offlineFlag = false;
		
		allFiles = new ArrayList<File>();
		returnFilePath(datasetPath);
		
		PrintWriter pw;
		double[] avgValues = new double[measures.size()];
		double[] offValues = new double[measures.size()];
		int[] numValues = new int[measures.size()];
		
		for(int i=0; i<allFiles.size(); i++){
			String path = allFiles.get(i).getPath();
			if(!path.contains("threshold") && !path.contains("snapshot") && !path.contains("BetaSeq") && !path.contains("ChangeLogs") && !path.contains("detailedMeasures")){
				String f = path.substring(0,path.lastIndexOf("/"));	//folder
				String meas = f.substring(f.lastIndexOf("/")+1, f.length());	//measure
				int ind = measures.indexOf(meas);
				
				BufferedReader bf = new BufferedReader(new FileReader(allFiles.get(i)));
				String line;
				if(!offline){
					while((line = bf.readLine())!= null){
						numValues[ind]++;
						avgValues[ind] += new Double(line);
					}
				}else{
					while((line = bf.readLine())!= null){
						StringTokenizer stg = new StringTokenizer(line, ",");
						String off = stg.nextToken().trim();
						if(off.length() > 0)
							offValues[ind] = new Double(off); //ignore offline value
						String val = stg.nextToken().trim();
						if(val.length() > 0){
							numValues[ind]++;
							avgValues[ind] += new Double(val);
						}
					}
				}
				
			}
		}
		
		//write averages in file
		File f = new File(datasetPath+"-Avg/");
		f.mkdirs();
		pw = new PrintWriter(new File(datasetPath+"-Avg/average.txt"));
		for(int k=0; k<measures.size(); k++){
			if(!measures.get(k).contains("thresholds") && !measures.get(k).contains("BetaSeq"))
				pw.print(measures.get(k).replaceAll(" ", "-")+" & ");
		}
		pw.println();
		
		for(int k=0; k<avgValues.length; k++){
			if(!measures.get(k).contains("thresholds") && !measures.get(k).contains("BetaSeq")){
				avgValues[k] = avgValues[k]/numValues[k];
				pw.print(String.format( "%.3f", avgValues[k]) + " & ");
			}
		}
		pw.println();
		if(offline){
			for(int k=0; k<offValues.length; k++){
				if(!measures.get(k).contains("thresholds") && !measures.get(k).contains("BetaSeq")){
					pw.print(String.format( "%.3f", offValues[k]) + " & ");
				}
			}
		}
		pw.close();
	
	}
	
	public void returnFilePath(String originalPath){
		File[] dir = new File(originalPath).listFiles();
		for(int i=0; i<dir.length; i++){
			String p = dir[i].getAbsolutePath();	//path
			if(!p.contains("thresholds") && !p.contains("snapshot") && !p.contains("misclass") && !p.contains("BetaSeq") && !p.contains("ChangeLogs") && !p.contains("detailedMeasures")){
				if(dir[i].isDirectory()){
					returnFilePath(dir[i].getAbsolutePath());
				}else{
						allFiles.add(dir[i]);
						String f = p.substring(0,p.lastIndexOf("/"));	//folder
						String m = f.substring(f.lastIndexOf("/")+1, f.length());	//measure
						if(!measures.contains(m))
							measures.add(m);	
				}
			}
		}
	}
}
