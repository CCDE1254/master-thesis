/**
 * 
 */
package master.thesis.randomnumber;

import info.quantlab.numericalmethods.lecture.randomnumbers.MersenneTwister;
import master.thesis.timediscretization.TimeDiscretization;

/**
 * @author QuanLiu
 *
 */

/*
 * this class used to generate random numbers
 */
public class MersenneTwisterSequence implements RandomNumber {

	int seed;
	double[] timeSeries;
	
	public MersenneTwisterSequence(int seed, double[] timeSeries) {
		super();
		this.seed = seed;
		this.timeSeries = timeSeries;
	}
	
	@Override
	public double[] getRandomNumberSeries() {
		MersenneTwister generator = new MersenneTwister(seed);
		double[] randomNumberSeries = new double[timeSeries.length];
		for(int i = 0; i < timeSeries.length; i++) {
			randomNumberSeries[i] = generator.nextDouble();
		}
		return randomNumberSeries;
	}


}
