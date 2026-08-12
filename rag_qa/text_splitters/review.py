class A():
    def a1(self):
        """?? a1 ???
        
        params:
            ??
        
        return:
            ??????"""
        print('你好A')
    def b1(self):
        """?? b1 ???
        
        params:
            ??
        
        return:
            ??????"""
        self.a1()

class B(A):
    def a1(self,):
        """?? a1 ???
        
        params:
            ??
        
        return:
            ??????"""
        print('你好B')

b = B()
b.b1()

