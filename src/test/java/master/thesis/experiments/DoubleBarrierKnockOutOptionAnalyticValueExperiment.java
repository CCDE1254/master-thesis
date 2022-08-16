/**
 * 
 */
package master.thesis.experiments;

import master.thesis.barrieroption.DoubleBarrierKnockOutCall;
import master.thesis.barrieroption.DoubleBarrierKnockOutPut;

/**
 * @author QuanLiu
 *
 */
public class DoubleBarrierKnockOutOptionAnalyticValueExperiment {
	public static void main(String[] args) {
	double initialStockPrice = 110.0;
	double riskFreeRate = 0.05;
	double volatilityTerm = 0.25;
	
	double maturity = 0.25;//set evaluation time to be 0
	double strike = 110.0;
	
	double upperBoundFactorB = 150.0;
	double upperBoundExponentialDelta1 = 0.05;
	double lowerBoundFactorA = 60.0;
	double lowerBoundExponentialDelta2 = 0.05;
	
	DoubleBarrierKnockOutCall call = new DoubleBarrierKnockOutCall(initialStockPrice, riskFreeRate, volatilityTerm,
			maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
			lowerBoundFactorA, lowerBoundExponentialDelta2);
	System.out.println(call.getAnalyticValue());
	
	DoubleBarrierKnockOutPut put = new DoubleBarrierKnockOutPut(initialStockPrice, riskFreeRate, volatilityTerm,
			maturity, strike, upperBoundFactorB, upperBoundExponentialDelta1,
			lowerBoundFactorA, lowerBoundExponentialDelta2);
	System.out.println(put.getAnalyticValue());
	}
}
