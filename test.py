n = 10

def test(n):
	dp = [0] * (n+1)
	if n == 1:
		return 1
	dp[1] = 1
	dp[2] = 2
	for i in range(10):
		dp[i] = dp[i-1] + dp[i-2]
	return dp[n]
if __name__ == '__main__':
	test(10)