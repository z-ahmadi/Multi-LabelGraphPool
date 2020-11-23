package single_labeled;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.StringTokenizer;

import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public class GraphicalManagement {
	public int maxConcepts;
	static Graph pool; 
	static ArrayList<Double> confidenceValues = new ArrayList<Double>();
	
	/**
	 * 
	 * @param args[0] dataset path
	 * 		  args[1] batch size
	 * 		  args[2] the path to the directory of output 
	 * 		  args[3] index of the confidence level
	 * 		  args[4] epsilon value
	 * 		  args[5] mode of classifying: weighted vote or single current concept {vote, current}  
	 * 		  args[6] class index {first, last}
	 * 		  args[7] features to consider {all, numeric}
	 * 		  args[8] similarity mode {statistical, yang}
	 * 		  args[9] similarity threshold (for Yang method)
	 * 		  args[10] NoCovariance
	 * 		  args[11] type of statistical test {normal, cov}
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	
	public static double[][] newLoadChiTable(){
		double[][] table = new double[0][0];
		try {
			BufferedReader bf = new BufferedReader(new FileReader("dataset/chiTable-200-20.txt"));
			
			//first line contains the confidence levels
			String line = bf.readLine();
			StringTokenizer stg = new StringTokenizer(line, ",");
			int col = stg.countTokens();
			int ind = 0;
			while(stg.hasMoreTokens()){
				confidenceValues.add(new Double(stg.nextToken()));
				ind++;
			}
			
			ArrayList<double[]> lines = new ArrayList<double[]>();
			
			//write values without the number of variables (the first token)
			while((line = bf.readLine()) != null){
				stg = new StringTokenizer(line, ",");
				double[] temp = new double[col];
				for(int k=0; k<col; k++){
					temp[k] = new Double(stg.nextToken());
				}
				lines.add(temp);
			}
			
			table = new double[lines.size()][col];
			for(int k=0; k<lines.size(); k++)
				table[k] = lines.get(k);
			
		} catch (FileNotFoundException e) {
			System.out.println("No such chi square file");
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return table;
	}
	
	public static double[][] loadChiTable(){
		double[][] table = new double[0][0];
		try {
			BufferedReader bf = new BufferedReader(new FileReader("dataset/chiTable.csv"));
			
			//first line contains the confidence levels
			String line = bf.readLine();
			StringTokenizer stg = new StringTokenizer(line, ",");
			int col = stg.countTokens();
			double[] confidence = new double[col];
			int ind = 0;
			while(stg.hasMoreTokens()){
				confidence[ind] = new Double(stg.nextToken());
				ind++;
			}
			
			ArrayList<double[]> lines = new ArrayList<double[]>();
			
			//write values without the number of variables (the first token)
			while((line = bf.readLine()) != null){
				stg = new StringTokenizer(line, ",");
				double[] temp = new double[col];
				stg.nextToken(); //ignore the number of line
				for(int k=0; k<col; k++){
					temp[k] = new Double(stg.nextToken());
				}
				lines.add(temp);
			}
			
			table = new double[lines.size()][col];
			for(int k=0; k<lines.size(); k++)
				table[k] = lines.get(k);
			
		} catch (FileNotFoundException e) {
			System.out.println("No such chi square file");
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return table;
	}
	
	public static void main(String[] args) throws Exception {
		double epsilon = Math.pow(10,-3);
		String path = args[2];
//		String info = args[2]+"/batchInfos"; //1
		File f = new File(path); //info		//2
		f.mkdirs();
		PrintWriter pw = new PrintWriter(new File(path+"/GraphPoolAcc.txt"));
		PrintWriter chw = new PrintWriter(new File(path+"/GraphPool-changeLog.txt"));
		PrintWriter pw2 = new PrintWriter(new File(path+"/GraphPoolAcc_summery.txt"));
		
		//keep the snapshot of pool in every batch for later presentations and plotting!
		ArrayList<ArrayList<double[][]>> meanOfAllConceptsInHistory = new ArrayList<ArrayList<double[][]>>(); 
				
		Instances dataSet = new Instances(new FileReader(args[0]));
		//for letter and gas-sensor dataset class index is 0: 
		if(args[6].equals("first"))
			dataSet.setClassIndex(0);
		else if(args[6].equals("last"))
			dataSet.setClassIndex(dataSet.numAttributes()-1);
		
		//preprocess on sensor data 
		ArrayList<Integer> numericIndeces =	preprocessDataset(args[0], dataSet, args[7]);
		int batchSize = new Integer(args[1]);

		double chiConf = new Double(args[3]);
		epsilon = new Double(args[4]);
		String mode = args[8];
		double simThresh = new Double(args[9]);
		String statType = args[11];
		String cvType = args[10]; 
		double chiValue = 0;
		//for the first chiTable load which was limited
//		double[][] chiTable = loadChiTable();
//		if(numericIndeces.size() <= 30){
//			chiValue = chiTable[numericIndeces.size()-1][chiIndex];
//			System.out.println("START!\nchisquare value = "+chiTable[numericIndeces.size()-1][chiIndex]);
//			pw2.println("START!\nchisquare value = "+chiTable[numericIndeces.size()-1][chiIndex]);
//		}else if(numericIndeces.size() <= 100){ 
//			int mod = numericIndeces.size()/10;
//			int indx = (mod - 3)+30;
//			chiValue = chiTable[indx-1][chiIndex];
//		}else{
//			System.err.println("Number of features are more than supported! (max features = 100)");
//		}
		
		//keep in mind that the degree of freedom is p(q-1), where p is the number of attributes and q is the number of populations.
		//and as we are comparing two populations, we only consider p!
		double[][] chiTable = newLoadChiTable();
		int chiIndex = confidenceValues.indexOf(chiConf);
		
		if(statType.equals("normal")){
			int cf = numericIndeces.size()*(numericIndeces.size()+3)/2;
			if(cf <= chiTable.length){ //it works for feature number <= 18
				chiValue = chiTable[cf-1][chiIndex];
				System.out.println("START!\nchisquare value = "+chiValue);
				pw2.println("START!\nchisquare value = "+chiValue);
			}else{
				chiValue = chiTable[chiTable.length-1][chiIndex];
				System.err.println("Number of features are more than supported! (max features = 18)");
			}
		}else if(statType.equals("cov")){
			if(numericIndeces.size() <= chiTable.length){
				chiValue = chiTable[numericIndeces.size()-1][chiIndex];
				System.out.println("START!\nchisquare value = "+chiValue);
				pw2.println("START!\nchisquare value = "+chiValue);
			}else{
				chiValue = chiTable[chiTable.length-1][chiIndex];
				System.err.println("Number of features are more than supported! (max features = 200)");
			}
		}else{
			System.err.println("type of statistical test is not specified!");
		}
		
		pool = new Graph(chiValue, pw, chw);
//		Evaluation eval = new Evaluation(dataSet);
		double avgAcc = 0, avgP = 0, avgR = 0, avgF = 0 , cumAcc = 0;
		
		long t1 = System.currentTimeMillis();
		
		pw.println("batch\tcumAcc\tAcc\tPrecision\tRecall\tF-measure\tVinPool");
		pw2.println("batch\tcumAcc\tAcc\tPrecision\tRecall\tF-measure\tVinPool");
		
		for(int i=0; i<Math.ceil((double)dataSet.numInstances()/batchSize); i++){
//			PrintWriter pwb = new PrintWriter(new File(info+"/"+i+".txt")); //print batch CVs' information	//3
			
			System.out.println("----"+i*batchSize+"----");
//			pw.write("----"+i+"----\n");
			Instances batch;
			if((i+1)*batchSize < dataSet.numInstances()){
				batch = new Instances(dataSet, i*batchSize, batchSize);
			}else{
				batch = new Instances(dataSet, i*batchSize, dataSet.numInstances()-i*batchSize);
			}
			
			//test of current batch
			if(i >= 1){
//				double error = 0;
//					for(int ind=0; ind<batch.size(); ind++){
//						if(batch.get(ind).classValue() != pool.classifyInstance(batch.get(ind)))
//							error += 1;
//					}
//				error = error/batch.size();
					
				Measurements currentM = pool.EvaluationOfBatch(new Evaluation(batch), batch, args[5]);
				cumAcc += currentM.Accuracy;
				avgAcc += currentM.Accuracy*batch.numInstances();
				avgP += currentM.Precision*batch.numInstances();
				avgR += currentM.Recall*batch.numInstances();
				avgF += currentM.F1*batch.numInstances();
				if(batch.size() < batchSize){
					pw.println(i+"\t"+avgAcc/(dataSet.numInstances()-batchSize)+"\t"+currentM.Accuracy+"\t"+currentM.Precision+"\t"+currentM.Recall+"\t"+currentM.F1+"\t"+pool.vertexes.size());
					pw2.println(i+"\t"+avgAcc/(dataSet.numInstances()-batchSize)+"\t"+currentM.Accuracy+"\t"+currentM.Precision+"\t"+currentM.Recall+"\t"+currentM.F1+"\t"+pool.vertexes.size());
					System.out.println(i+"\t"+avgAcc/(dataSet.numInstances()-batchSize)+"\t"+currentM.Accuracy+"\t"+currentM.Precision+"\t"+currentM.Recall+"\t"+currentM.F1+"\t"+pool.vertexes.size());
				}else{
					pw.println(i+"\t"+avgAcc/(i*batchSize)+"\t"+currentM.Accuracy+"\t"+currentM.Precision+"\t"+currentM.Recall+"\t"+currentM.F1+"\t"+pool.vertexes.size());
					pw2.println(i+"\t"+avgAcc/(i*batchSize)+"\t"+currentM.Accuracy+"\t"+currentM.Precision+"\t"+currentM.Recall+"\t"+currentM.F1+"\t"+pool.vertexes.size());
					System.out.println(i+"\t"+avgAcc/(i*batchSize)+"\t"+currentM.Accuracy+"\t"+currentM.Precision+"\t"+currentM.Recall+"\t"+currentM.F1+"\t"+pool.vertexes.size());
				}
			}
			
			//train model
			//add the new data to the graph, set the edges,...
			pool.addVertex(batch, numericIndeces, i, epsilon, mode, statType, simThresh, cvType);
			
			//add the snapshot of concept means to arraylist
			ArrayList<double[][]> curConceptMeans = new ArrayList<double[][]>();
			Iterator<Vertex> iter = pool.vertexes.iterator();
			while(iter.hasNext()){
				double[][] mv = iter.next().getCV().getMeanVector();
				double[][] mv2 = new double[mv.length+1][mv[0].length];
				for(int ic=0; ic<mv2[0].length; ic++){
					for(int ir=0; ir<mv.length;ir++){
						mv2[ir][ic] = mv[ir][ic];
					}
					mv2[mv.length][ic] = ic;
				}
				curConceptMeans.add(mv2);
			}
			meanOfAllConceptsInHistory.add(curConceptMeans);
			
//			pool.print(pwb);	//4
//			pwb.close();		//5
			//for hyperplane visualization
			/*
			
			//write adjustancy matrix of this snapshot
			int numV = pool.vertexes.size();
			double[][] adjustMat = new double[numV][numV];
			for(int id=0; id<numV; id++){
				Vertex v = pool.vertexes.get(id);
				ArrayList<Vertex> NV = v.neighbors;
				ArrayList<Double> wV = v.TransitionWeights;
				for(int jd=0; jd<NV.size(); jd++){
					int jInd = indexOf(NV.get(jd), pool.vertexes);
					adjustMat[id][jInd] = wV.get(jd);
				}
			}
			PrintWriter pwam = new PrintWriter(new File(path+"/adjustMatrix/"+(i+1)+".txt"));
			for(int id=0; id<numV; id++){
				pwam.print("{");
				for(int jd=0; jd<numV; jd++){
					if(jd<numV-1)
						pwam.print(new DecimalFormat("#.##").format(adjustMat[id][jd])+",");
					else
						pwam.print(new DecimalFormat("#.##").format(adjustMat[id][jd])+"},");
				}
//				pwam.println();
			}
			pwam.close();
			*/
			
			
			pw.write("-------------------\n");
			
//			AttrInfo[] infos = cr.attrInfos;
			
			
			//for hyperplane visualization
//			pw.print(i + " ");
//			System.out.println(infos.length);
//			for(int j=0; j< infos.length; j++){
//				if(infos[j].isNominal){
//					for(int k=0; k<infos[j].classAttr.numValues(); k++){
//						for(int h=0; h<infos[j].attr.numValues(); h++)
//							pw.print(infos[j].probs[k][h]+" ");
//						pw.print(" | ");
//					}
//					pw.print(" || ");
//				}else{
//					for(int k=0; k<infos[j].classAttr.numValues(); k++)
//						pw.print(infos[j].miu[k]+" , "+ infos[j].sigma[k]+" ; ");
//					pw.print(" || "); 
//				}
//			}
//			pw.println();
		}
		
		long t2 = System.currentTimeMillis();
		avgAcc = avgAcc / (dataSet.numInstances()-batchSize);
		avgP = avgP / (dataSet.numInstances()-batchSize);
		avgR = avgR / (dataSet.numInstances()-batchSize);
		avgF = avgF / (dataSet.numInstances()-batchSize);
		
		System.out.println("*** Chi = "+ chiValue+" ***");
		System.out.println("Average Accuracy = "+avgAcc+"\nAverage Precision = "+avgP+"\nAverage Recall = "+avgR+"\nAverage F = "+avgF);
		System.out.println("total time = "+(t2-t1));
		System.out.println("pool size = "+pool.vertexes.size());
		pw2.println("\n\nAverage Accuracy = "+avgAcc+"\nAverage Precision = "+avgP+"\nAverage Recall = "+avgR+"\nAverage F = "+avgF);
		pw2.println("total time = "+(t2-t1));
		
		
		/*
		for(int i=0; i<meanOfAllConceptsInHistory.size(); i++){
			PrintWriter pwm = new PrintWriter(new File(args[2]+"/means/ConceptMeans_"+(i+1)+".txt"));
			ArrayList<double[][]> cmean = meanOfAllConceptsInHistory.get(i);
			String strBatch = "";
			for(int j=0; j<cmean.size(); j++){
				strBatch = strBatch.concat(arrayToString(cmean.get(j)));
			}
			pwm.print(strBatch.trim());
//			pwm.println("["+strBatch.substring(0, strBatch.length()-1)+"]");
			pwm.close();
		}
		*/
		
		pw.close();
		pw2.close();
		chw.close();
	}
	
	
	public static int indexOf(Vertex v, ArrayList<Vertex> av){
		int indx = -1;
		for(int i=0; i<av.size();i++){
			if(av.get(i).equals(v)){
				indx = i;
				break;
			}
		}
		
		return indx;
	}
	
	
	/**
	 * write the transpose of a into string (for Matlab use)
	 * @param a
	 * @return
	 */
	public static String arrayToString(double[][] a){
		String ret = "";
		for(int j=0; j<a[0].length; j++){
			for(int i=0; i<a.length; i++){
				ret = ret.concat(a[i][j]+",");
			}
			ret = ret.substring(0, ret.length()-1).concat("\n");
		}
		return ret;
	}

	
	static ArrayList<Integer> preprocessDataset(String name, Instances dataSet, String typeOfFeatures){
//		if(name.contains("sensor"))
//			for (int i = 0; i < dataSet.numInstances(); i++) 
//				dataSet.instance(i).setValue(0, dataSet.instance(i).value(0) / 60);
		
		ArrayList<Integer> numericInd = new ArrayList<Integer>();
		//separate nominal and numeric attributes 
		for(int i=0; i<dataSet.numAttributes(); i++){
			if(typeOfFeatures.equals("all") && i!=dataSet.classIndex())
				numericInd.add(i);
			else if(typeOfFeatures.equals("numeric")){
				if(dataSet.attribute(i).isNumeric()){
					numericInd.add(i);
				}				
			}
		}
		
		return numericInd;
	}

}
