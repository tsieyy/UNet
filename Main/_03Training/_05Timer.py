
import time

def timer(long = -1):
	start = time.time()
	def end(method_name="Unnamed function"):
		print(method_name + " took : " + str(time.time() - start)[0:long] + " seconds.")
		return

	return end

class Timer():
	def __init__(self,long = 6):
		self.long = long
		self.start = time.time()
	def __str__(self):      # 魔法方法 print实现时间打印
		return ("Time: " + str(time.time() - self.start)[0:self.long] + " seconds.")

#函数运行时间 装饰器
def time_cost(func):
	def cost():
		start = time.time()
		func()
		print(func.__name__ + ' method cost: ' + str(time.time() - start)[0:-1] + ' seconds')
	return cost


@time_cost
def test():
	for i in range(10000):
		print(1)

if __name__ == '__main__':
	# end = timer(long=8)
	# time.sleep(1)
	# end("Test")
	#
	# a = Timer(8)
	# time.sleep(1)
	# print(a)
	# time.sleep(1)
	# print(a)
	test()

