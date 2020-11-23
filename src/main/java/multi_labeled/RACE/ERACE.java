package multi_labeled.RACE;

import weka.classifiers.UpdateableClassifier;

public class ERACE {
	/**
     * The number of classifier chain models
     */
    protected int numOfModels;

    /**
     * An array of RACE models
     */
    protected LRUpdateable[] ensemble;

}
