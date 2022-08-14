/**
 * 
 */
package master.thesis.brownianmotion;

import net.finmath.functions.NormalDistribution;

/**
 * @author QuanLiu
 *
 */

/*
 * this class used to generate 
 */

public class BrownianMotionGenerator implements BrownianMotion {

	double[] timeSeries;
	double[] randomNumberSequence;

	public BrownianMotionGenerator(double[] timeSeries, double[] randomNumberSequence) {
		super();
		this.timeSeries = timeSeries;
		this.randomNumberSequence = randomNumberSequence;
	}

	@Override
	public double getNormalDistributedNumber(double randomNumber) {
		double normal = NormalDistribution.inverseCumulativeDistribution(randomNumber);
		return normal;
	}
	
	@Override
	public double[] getBrownianMotionPath() {
		double[] brownianMotionPath = new double[timeSeries.length];
		brownianMotionPath[0] = 0.0;
		for(int i = 1; i < timeSeries.length; i++) {
			brownianMotionPath[i] = brownianMotionPath[i-1] + Math.sqrt(timeSeries[i] - timeSeries[i-1]) * getNormalDistributedNumber(randomNumberSequence[i-1]);
		}
		return brownianMotionPath;
	}

}
