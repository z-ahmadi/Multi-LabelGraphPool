package multi_labeled;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.StringTokenizer;

public class generateArffFromPYPoutput {

	public static void main(String[] args) throws IOException {
		String path = "/home/havij/Dropbox/Scripts-paper codes/Multi-labelGraphPool/testPYP_0.2_0.8_0.5_n3000_t20_d0.75-shuffled";
		int numWords = 100, numTopics = 100; 
		
		BufferedReader bf = new BufferedReader(new FileReader(path));
		String line;
		
		PrintWriter pw = new PrintWriter(new File(path+".arff"));
		PrintWriter pwn = new PrintWriter(new File(path+"-numbers.arff"));
		pw.print("@relation syntPYP\n\n");
		pwn.print("@relation syntPYP\n\n");
		for(int i=0; i<numWords; i++) {
			pw.println("@attribute w"+i+" {0,1}");
			pwn.println("@attribute w"+i+" numeric");
		}
		for(int i=0; i<numTopics; i++) {
			pw.println("@attribute t"+i+" {0,1}");
			pwn.println("@attribute t"+i+" {0,1}");
		}
		pw.println("\n@data\n");
		pwn.println("\n@data\n");
		
		while((line = bf.readLine())!= null) {
			StringTokenizer stg = new StringTokenizer(line, "\t"); //labels and features are separated by tab
			stg.nextToken(); //instance number
			
			String topics = stg.nextToken();
			int[] topArr = new int[numTopics];
			StringTokenizer stgT = new StringTokenizer(topics, " ");
			while(stgT.hasMoreTokens()) {
				String t = stgT.nextToken();
				topArr[new Integer(t.substring(1))] = 1;
			}
			
			String words = stg.nextToken(); 
			int[] wordBArr = new int[numWords];
			int[] wordArr = new int[numWords];
			StringTokenizer stgW = new StringTokenizer(words, " ");
			while(stgW.hasMoreTokens()) {
				String t = stgW.nextToken();
				wordBArr[new Integer(t.substring(1))] = 1;
				wordArr[new Integer(t.substring(1))] += 1;
			}
			
			for(int i=0; i<numWords; i++) {
				pw.print(wordBArr[i]+",");
				pwn.print(wordArr[i]+",");
			}
			for(int i=0; i<numTopics-1; i++) {
				pw.print(topArr[i]+",");
				pwn.print(topArr[i]+",");
			}
			pw.print(topArr[numTopics-1]+"\n");
			pwn.print(topArr[numTopics-1]+"\n");
		}
		
		//generate xml file
		PrintWriter xml = new PrintWriter(new File(path+".xml"));
		xml.println("<?xml version=\"1.0\" encoding=\"utf-8\"?>");
		xml.println("<labels xmlns=\"http://mulan.sourceforge.net/labels\">");
		for(int i=0; i<numTopics; i++)
			xml.println(" <label name=\"t"+i+"\"></label>");
		xml.println("</labels>");

		xml.close();	
		pw.close();
		pwn.close();
	}

}
