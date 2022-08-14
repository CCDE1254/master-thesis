/**
 * 
 */
package master.thesis.brownianmotion;

/**
 * @author QuanLiu
 *
 */

/*
 * this interface used to generate brownian motion by given random number sequence
 */
public interface BrownianMotion {
	
	public double getNormalDistributedNumber(double randomNumber);
	
	public double[] getBrownianMotionPath();

}
