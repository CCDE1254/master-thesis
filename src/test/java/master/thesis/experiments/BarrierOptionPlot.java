/**
 * 
 */
package master.thesis.experiments;

import java.util.function.DoubleUnaryOperator;

import master.thesis.brownianmotion.BrownianMotion;
import master.thesis.brownianmotion.BrownianMotionGenerator;
import master.thesis.randomnumber.MersenneTwisterSequence;
import master.thesis.randomnumber.RandomNumber;
import master.thesis.timediscretization.TimeDiscretization;
import master.thesis.timediscretization.TimeDiscretizationWithEqualTimeStepSize;
import net.finmath.plots.Plot2D;
import net.finmath.plots.PlotProcess2D;
import net.finmath.plots.Plots;


/**
 * @author QuanLiu
 *
 */
public class BarrierOptionPlot {

	static int seed = 129;
	static double stratTime = 0.0;
	static double endTime = 10.0;
	static int numberOfTimeSteps = 100;
	
	double upperBoundFactorB = 1.2;
	double upperBoundExponentialDelta = 0.5;
	double lowerBoundFactorB = 1.2;
	double lowerBoundExponentialDelta = 0.5;
	int numberOfPath = 200;
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		//generate array of time points
		TimeDiscretization time = new TimeDiscretizationWithEqualTimeStepSize(stratTime, endTime, numberOfTimeSteps);
		double[] timeSeries = time.getTimeSeries();
		
		//generate random number sequence under given time series
		RandomNumber randomnumber = new MersenneTwisterSequence(seed, timeSeries);
		double[] sequenceOfRandomNumbers = randomnumber.getRandomNumberSeries();
		
		//generate brownian motion under given time series and random number sequence
		BrownianMotion brownianMotion = new BrownianMotionGenerator(timeSeries, sequenceOfRandomNumbers);
		double[] brownianMotionPath = brownianMotion.getBrownianMotionPath();


	    DoubleUnaryOperator trajectory = t -> {
			return (brownianMotionPath[(int) t]);
		};

			Plot2D plot = new Plot2D(0, numberOfTimeSteps, numberOfTimeSteps + 1, trajectory);
			plot.setTitle("Brownian motion path");
			plot.setXAxisLabel("Time");
			plot.setYAxisLabel("Brownian motion");
			plot.show();

		
		

	}

}
