import math
from decimal import Decimal, getcontext
import numpy as np
import scipy.stats as stats

# Set high precision for handling large numbers
getcontext().prec = 100  # Adjust precision as needed

class ProbabilityCalculator:
    def __init__(self, total_outcomes: int):
        """
        Initializes the probability calculator with the total possible outcomes.
        
        Args:
            total_outcomes (int): Total number of possible outcomes.
        """
        if total_outcomes <= 0:
            raise ValueError("Total outcomes must be greater than zero.")
        self.total_outcomes = Decimal(total_outcomes)

    def calculate_probability(self, successful_outcomes: int) -> Decimal:
        """
        Calculates the probability of a successful outcome.
        
        Args:
            successful_outcomes (int): The number of successful outcomes.
        
        Returns:
            Decimal: Probability of the successful outcome.
        """
        if successful_outcomes < 0:
            raise ValueError("Successful outcomes must be zero or greater.")
        if successful_outcomes > self.total_outcomes:
            raise ValueError("Successful outcomes must not exceed total outcomes.")
        
        return Decimal(successful_outcomes) / self.total_outcomes

    def calculate_cumulative_probability(self, start: int, end: int) -> Decimal:
        """
        Calculates the cumulative probability for a range of successful outcomes.
        
        Args:
            start (int): Starting number of successful outcomes.
            end (int): Ending number of successful outcomes.
        
        Returns:
            Decimal: Cumulative probability for the range.
        """
        if start < 0 or end < 0:
            raise ValueError("Outcomes must be zero or greater.")
        if start > end or end > self.total_outcomes:
            raise ValueError("Invalid range.")
        
        cumulative_successes = sum(range(start, end + 1))
        return Decimal(cumulative_successes) / self.total_outcomes

    def calculate_joint_probability(self, probabilities: list) -> Decimal:
        """
        Calculates the joint probability for multiple independent events.
        
        Args:
            probabilities (list): List of probabilities for independent events.
        
        Returns:
            Decimal: Joint probability of all events occurring.
        """
        if any(prob < 0 or prob > 1 for prob in probabilities):
            raise ValueError("All probabilities must be in the range [0, 1].")
        
        joint_probability = Decimal(1)
        for prob in probabilities:
            joint_probability *= prob
        return joint_probability

    def calculate_conditional_probability(self, prob_a: Decimal, prob_b: Decimal, prob_a_given_b: Decimal) -> Decimal:
        """
        Calculates the conditional probability P(A|B).
        
        Args:
            prob_a (Decimal): Probability of event A.
            prob_b (Decimal): Probability of event B.
            prob_a_given_b (Decimal): Probability of event A given B.
        
        Returns:
            Decimal: Conditional probability P(A|B).
        """
        if prob_b <= 0:
            raise ValueError("Probability of event B must be non-zero.")
        
        return prob_a_given_b / prob_b

    def calculate_union_probability(self, prob_a: Decimal, prob_b: Decimal, prob_a_and_b: Decimal) -> Decimal:
        """
        Calculates the probability of the union of two events P(A ∪ B).
        
        Args:
            prob_a (Decimal): Probability of event A.
            prob_b (Decimal): Probability of event B.
            prob_a_and_b (Decimal): Probability of both events A and B occurring.
        
        Returns:
            Decimal: Probability of the union of events A and B.
        """
        return prob_a + prob_b - prob_a_and_b

    def store_large_number(self, exponent: int) -> Decimal:
        """
        Stores a large number represented by 10^exponent as a Decimal for high precision.
        
        Args:
            exponent (int): The exponent of the large number to store (e.g., for googolplex 10^100).
        
        Returns:
            Decimal: The large number 10^exponent as a Decimal.
        """
        if exponent < 0:
            raise ValueError("Exponent must be non-negative.")
        # Storing the large number 10^exponent
        return Decimal(10) ** exponent

    # Additional probability distributions
    def normal_distribution(self, x: float, mean: float, std_dev: float) -> Decimal:
        """
        Calculates the probability density function for a normal distribution.
        
        Args:
            x (float): The value for which to calculate the probability density.
            mean (float): The mean of the distribution.
            std_dev (float): The standard deviation of the distribution.
        
        Returns:
            Decimal: The probability density at x.
        """
        pdf_value = stats.norm.pdf(x, loc=mean, scale=std_dev)
        return Decimal(pdf_value)

    def binomial_distribution(self, n: int, k: int, p: float) -> Decimal:
        """
        Calculates the probability of exactly k successes in n trials for a binomial distribution.
        
        Args:
            n (int): Number of trials.
            k (int): Number of successful outcomes.
            p (float): Probability of success on an individual trial.
        
        Returns:
            Decimal: Probability of exactly k successes in n trials.
        """
        if n < 0 or k < 0 or p < 0 or p > 1:
            raise ValueError("Input parameters must be valid.")
        
        binom_coeff = math.comb(n, k)
        prob = binom_coeff * (p ** k) * ((1 - p) ** (n - k))
        return Decimal(prob)

# Example usage
if __name__ == "__main__":
    googolplex_exponent = 10 ** 100  # Exponent for Googolplex (10^10^100)
    calculator = ProbabilityCalculator(10 ** 100)  # Set total outcomes for Googol (10^100)

    # Store and print large number (googolplex)
    large_number = calculator.store_large_number(googolplex_exponent)
    print("Stored large number (Googolplex):", large_number)

    # Calculate specific probability
    probability = calculator.calculate_probability(1)
    print("Probability of a single successful outcome:", probability)

    # Calculate cumulative probability for a range of outcomes
    cumulative_probability = calculator.calculate_cumulative_probability(1, 1000)
    print("Cumulative probability for the first 1000 successful outcomes:", cumulative_probability)

    # Calculate joint probability for multiple independent events
    event_probabilities = [calculator.calculate_probability(1), calculator.calculate_probability(2)]
    joint_probability = calculator.calculate_joint_probability(event_probabilities)
    print("Joint probability for multiple events:", joint_probability)

    # Calculate conditional probability P(A|B)
    prob_a = calculator.calculate_probability(10)  # Example probability of A
    prob_b = calculator.calculate_probability(5)   # Example probability of B
    prob_a_given_b = calculator.calculate_probability(2)  # Probability of A given B
    conditional_probability = calculator.calculate_conditional_probability(prob_a, prob_b, prob_a_given_b)
    print("Conditional probability P(A|B):", conditional_probability)

    # Calculate probability for a normal distribution
    normal_prob = calculator.normal_distribution(0, 0, 1)  # Normal distribution with mean 0 and std dev 1
    print("Normal distribution probability at 0:", normal_prob)

    # Calculate probability for a binomial distribution
    binomial_prob = calculator.binomial_distribution(10, 3, 0.5)  # 10 trials, 3 successes, success probability 0.5
    print("Binomial distribution probability:", binomial_prob)