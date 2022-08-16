/**
 * 
 */
package master.thesis.barrieroption;

import net.finmath.functions.NormalDistribution;

/**
 * @author QuanLiu
 *
 */
public class DoubleBarrierKnockOutPut {
	double initialStockPrice;
	double riskFreeRate;
	double volatilityTerm;
	
	double maturity;//set evaluation time to be 0
	double strike;
	
	double upperBoundFactorB;
	double upperBoundExponentialDelta1;
	double lowerBoundFactorA;
	double lowerBoundExponentialDelta2;
	
	
	public DoubleBarrierKnockOutPut(double initialStockPrice, double riskFreeRate, double volatilityTerm,
			double maturity, double strike, double upperBoundFactorB, double upperBoundExponentialDelta1,
			double lowerBoundFactorA, double lowerBoundExponentialDelta1) {
		super();
		this.initialStockPrice = initialStockPrice;
		this.riskFreeRate = riskFreeRate;
		this.volatilityTerm = volatilityTerm;
		this.maturity = maturity;
		this.strike = strike;
		this.upperBoundFactorB = upperBoundFactorB;
		this.upperBoundExponentialDelta1 = upperBoundExponentialDelta1;
		this.lowerBoundFactorA = lowerBoundFactorA;
		this.lowerBoundExponentialDelta2 = lowerBoundExponentialDelta1;
	}
	
	public double getAnalyticValue() {
		double sumPart1 = 0.0;
		double sumPart2 = 0.0;
		for(double n = -10.0; n <= 10.0; n++) {//here let n loop from -10 to 10
			double c1n = 2.0*(riskFreeRate - lowerBoundExponentialDelta2 - n*(upperBoundExponentialDelta1 - lowerBoundExponentialDelta2))/Math.pow(volatilityTerm, 2.0) + 1.0;
			double c2n = 2.0*n*(upperBoundExponentialDelta1 - lowerBoundExponentialDelta2)/Math.pow(volatilityTerm, 2.0);
			double c3n = 2.0*(riskFreeRate - lowerBoundExponentialDelta2 + n*(upperBoundExponentialDelta1 - lowerBoundExponentialDelta2))/Math.pow(volatilityTerm, 2.0) + 1.0;
			double F = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*maturity);
			double tau = maturity;
			double d1n = ( Math.log(initialStockPrice*Math.pow(upperBoundFactorB, 2.0*n)/(F*Math.pow(lowerBoundFactorA, 2.0*n))) + (riskFreeRate + Math.pow(volatilityTerm, 2.0)/2.0)*tau )
					/(volatilityTerm*Math.sqrt(tau));
			double d2n = ( Math.log(initialStockPrice*Math.pow(upperBoundFactorB, 2.0*n)/(strike*Math.pow(lowerBoundFactorA, 2.0*n))) + (riskFreeRate + Math.pow(volatilityTerm, 2.0)/2.0)*tau )
					/(volatilityTerm*Math.sqrt(tau));
			double d3n = ( Math.log(Math.pow(lowerBoundFactorA, 2.0*n+2.0)/(F*initialStockPrice*Math.pow(upperBoundFactorB, 2.0*n))) + (riskFreeRate + Math.pow(volatilityTerm, 2.0)/2.0)*tau )
					/(volatilityTerm*Math.sqrt(tau));
			double d4n = ( Math.log(Math.pow(lowerBoundFactorA, 2.0*n+2.0)/(strike*initialStockPrice*Math.pow(upperBoundFactorB, 2.0*n))) + (riskFreeRate + Math.pow(volatilityTerm, 2.0)/2.0)*tau )
					/(volatilityTerm*Math.sqrt(tau));
			sumPart1 += Math.pow(Math.pow(upperBoundFactorB, n)/Math.pow(lowerBoundFactorA, n), c1n)*Math.pow(lowerBoundFactorA/initialStockPrice, c2n)
					*(NormalDistribution.cumulativeDistribution(d1n) - NormalDistribution.cumulativeDistribution(d2n))
					-Math.pow(Math.pow(lowerBoundFactorA, n+1.0)/(Math.pow(upperBoundFactorB, n)*initialStockPrice), c3n)
					*(NormalDistribution.cumulativeDistribution(d3n) - NormalDistribution.cumulativeDistribution(d4n));
			sumPart2 += Math.pow(Math.pow(upperBoundFactorB, n)/Math.pow(lowerBoundFactorA, n), c1n - 2.0)*Math.pow(lowerBoundFactorA/initialStockPrice, c2n)
					*(NormalDistribution.cumulativeDistribution(d1n - volatilityTerm*Math.sqrt(tau)) - NormalDistribution.cumulativeDistribution(d2n - volatilityTerm*Math.sqrt(tau)))
					-Math.pow(Math.pow(lowerBoundFactorA, n+1.0)/(Math.pow(upperBoundFactorB, n)*initialStockPrice), c3n - 2.0)
					*(NormalDistribution.cumulativeDistribution(d3n - volatilityTerm*Math.sqrt(tau)) - NormalDistribution.cumulativeDistribution(d4n - volatilityTerm*Math.sqrt(tau)));
		}
		double putPrice = -initialStockPrice * sumPart1 + strike * Math.exp(-riskFreeRate*maturity) * sumPart2;
		return putPrice;
	}
	
}
