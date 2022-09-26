/**
 * 
 */
package master.thesis.experiments;

import java.text.DecimalFormat;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

import master.thesis.importance.sampling.DoubleBarrierKnockOutCallImportanceSampling;
import master.thesis.neural.network.OptimalParameterFinderISCall;
import master.thesis.randomnumber.MersenneTwisterSequence;
import master.thesis.timediscretization.TimeDiscretizationWithEqualTimeStepSize;
import net.finmath.plots.Plot;
import net.finmath.plots.Plot3DFX;
import net.finmath.plots.Plots;


/**
 * @author QuanLiu
 *
 */
public class ImportanceSamplingPossibleValueSurfacePlot {
	static int numberOfSimulations = 50;
	static double stratTime = 0.0;
	static double endTime = 10.0;
	static int numberOfTimeSteps = 1000;
	static double initialStockPrice = 100;
	static double strike = 150;
	static double riskFreeRate = 0.02;
	static double volatilityTerm = 0.15;
	
	static double upperBoundFactorB = 150;
	static double upperBoundExponentialDelta1 = 0.06;
	static double lowerBoundFactorA = 50;
	static double lowerBoundExponentialDelta2 = 0.06;
    static int numberOfPossibleValues = 10;
    static double minEta1 = -0.02;
	static double maxEta1 = 0.02;
	static double minEta2 = -1.0/1.95;
	static double maxEta2 = -1.0/2.05;
	
	public static void main(String[] ards) throws Exception {
		DoubleBinaryOperator trajectory = (eta1, eta2) -> {
			return (getValue(eta1, eta2));
		};
		Plot3DFX plot1 = new Plot3DFX(minEta1, maxEta1, minEta2, maxEta2, numberOfPossibleValues, numberOfPossibleValues, trajectory);

		plot1.setIsLegendVisible(true).setTitle("surface").setXAxisLabel("eta1").setYAxisLabel("eta2").setZAxisLabel("standard error").show();
	}
		
	 
	public static double getValue(double eta1, double eta2) {
		
		double[] eta = {eta1,eta2};
		
		//generate array of time points
    	TimeDiscretizationWithEqualTimeStepSize time = new TimeDiscretizationWithEqualTimeStepSize(stratTime, endTime, numberOfTimeSteps);
		double[] timeSeries = time.getTimeSeries();
		
		//generate random number sequence under given time series
		MersenneTwisterSequence randomnumber = new MersenneTwisterSequence(numberOfSimulations, timeSeries);
		double[][] randomNumberMatrix = randomnumber.getRandomNumberRealizations();
		DoubleBarrierKnockOutCallImportanceSampling callIS = new DoubleBarrierKnockOutCallImportanceSampling(numberOfSimulations, numberOfTimeSteps,
						randomNumberMatrix,
						initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
						endTime, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
						lowerBoundExponentialDelta2, eta);
		return callIS.getStandardError();
	}

}
