package single_labeled;


import moa.classifiers.AbstractClassifier;
import weka.core.Instances;

public class Measurements {
	public double Accuracy, Precision, Recall, F1;
	int TP = 0, TN = 0, FP = 0, FN = 0;
	
//	public Measurements(AbstractClassifier c, Instances inst){
//		for(int i=0; i<inst.numInstances(); i++){
//			if(c.correctlyClassifies(inst.get(i))){
//				if((int)inst.get(i).classValue() == )
//					//if multi-class???
//			}else{
//				
//			}
//		}
//	}

}
