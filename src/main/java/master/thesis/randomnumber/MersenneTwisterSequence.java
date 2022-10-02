/**
 * 
 */
package master.thesis.randomnumber;

import info.quantlab.numericalmethods.lecture.randomnumbers.MersenneTwister;

/**
 * @author QuanLiu
 *
 */

/*
 * this class used to generate random numbers
 */
public class MersenneTwisterSequence {

	int numberOfSimulations;
	double[] timeSeries;
	
	
	public MersenneTwisterSequence(int numberOfSimulations, double[] timeSeries) {
		super();
		this.numberOfSimulations = numberOfSimulations;
		this.timeSeries = timeSeries;
	}



	public double[][] getRandomNumberRealizations() {
		
		double[][] randomNumberMatrix = new double[timeSeries.length-1][numberOfSimulations];
		for(int j = 0; j < numberOfSimulations; j++) {
			MersenneTwister generator = new MersenneTwister(j+1);
			for(int i = 0; i < timeSeries.length-1; i++) {
				randomNumberMatrix[i][j] = generator.nextDouble();
			}
		}
		return randomNumberMatrix;
	}


}
