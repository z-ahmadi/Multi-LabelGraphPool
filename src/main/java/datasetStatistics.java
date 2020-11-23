import mulan.core.MulanException;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.data.Statistics;

public class datasetStatistics {

	public static void main(String[] args) throws MulanException {
		//String path = "/home/potato/Dropbox/Backup Codes/multi-labelGraphpool/data/testPYP_0.2_0.8_n1000_recur_t20_d0.75_exactRepeat";
		String path = "/home/potato/Dropbox/Backup Codes/multi-labelGraphpool/data/testPYP_0.2_0.8_0.5_n3000_t20_d0.75-shuffled";

		
		MultiLabelInstances dataset = new MultiLabelInstances(path+".arff", path+".xml");
		Statistics stat = new Statistics();
		
		stat.calculateStats(dataset);
		
		System.out.println("cardinality = "+stat.cardinality());
		System.out.println("density = "+stat.density());
		System.out.println("unique labelset = "+stat.labelSets().size());

	}

}
