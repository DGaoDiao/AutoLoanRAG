class A():
    def a1(self):
        """执行 a1 函数。
                
                params:
                    无。
                
                return:
                    函数返回值。"""
        print('你好A')
    def b1(self):
        """执行 b1 函数。
                
                params:
                    无。
                
                return:
                    函数返回值。"""
        self.a1()

class B(A):
    def a1(self,):
        """执行 a1 函数。
                
                params:
                    无。
                
                return:
                    函数返回值。"""
        print('你好B')

b = B()
b.b1()

