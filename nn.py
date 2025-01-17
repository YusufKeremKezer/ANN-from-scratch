from abc import ABC,abstractmethod
import numpy as np

class NN(ABC):
    @abstractmethod
    def forward(self,x):
        pass

    @abstractmethod
    def backward(self,x):
        pass




class Layer(NN):
    def __init__(self,dim1,dim2,initialization="random"):
        initializer = Initializer()
        self.W, self.b = initializer.initialize(dim1, dim2,initialization)
        self.dW=None
        self.db=None
        self.X=None
        
    def forward(self,X):
        # Input features = 20, Neurons = 10
        self.X = X
        z = np.dot(self.X,self.W) + self.b
        return z

    def backward(self,grad):
        # derivative of layer      
        self.dW = np.dot(self.X.T,grad)
        self.db = np.sum(grad,axis=0,keepdims=True)
        # derivate of input
        self.input=np.dot(grad,self.W.T)
        return self.input





class ReLU(NN):
    
    def forward(self,x):
        self.out = np.maximum(x,0)
        return self.out

    def backward(self,x):
        return x*(self.out>0)


class Sigmoid(NN):
    
    def forward(self,x):
        self.out = 1/(1+np.exp(-x))
        return self.out

    def backward(self,grad):
        return grad*(1-self.out)*self.out
    





class BCELoss(NN):
            

    def forward(self,y_hat,y):
        
        epsilon=1e-10
        
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        
        return np.sum(-y*np.log(y_hat)-(1-y)*np.log(1-y_hat))

    def backward(self,y_hat,y):
        return y_hat-y
    
    
class NeuralNetwork(NN):
    def __init__(self,layers,loss):
        self.layers=layers
        self.grad_layers=[]
        self.y_hat=None
        self.y=None
        self.loss = loss
        self.loss_value=None
        
    def forward(self,x,y):
        
        for layer in self.layers:
            x=layer.forward(x)

        self.y_hat=x
        self.y=y
        
        self.loss_value = self.loss.forward(self.y_hat,self.y)

        return self.loss_value
    
    def backward(self):
        
        self.x = self.loss.backward(self.y_hat,self.y)

        for layer in reversed(self.layers):
            if isinstance(layer,Layer):
                self.grad_layers.append(layer)
                
            self.x=layer.backward(self.x)
            
        return self.grad_layers




class Optimizer(ABC):
    @abstractmethod
    def __init__(self,lr):
        pass
    
    @abstractmethod
    def step(self,layers):
        pass

class GradientDescent(Optimizer):
    
    def __init__(self, lr):
        self.lr = lr
    
    def step(self, layers):
        for layer in layers:
            if isinstance(layer, Layer):  
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db
                
                
                
class Initializer:
    _instance=None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Initializer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        
        
        self.init_dict = { 
            "random": self.random_initialization,  # Initialization types stored like this because of Open Closed principle
            "he": self.he_initialization,  
        }

    def initialize(self, dim1, dim2, initialization):
        # Call the initialization function stored in init_dict
        if initialization not in self.init_dict:
            raise KeyError(f"Initialization method '{initialization}' not found in init_dict.")
        
        self.method = self.init_dict[initialization](dim1, dim2)
        return self.method

    def random_initialization(self, dim1, dim2):
        W = np.random.randn(dim1, dim2) * 0.01  
        b = np.zeros(shape=( 1,dim2))  
        return W, b

    def he_initialization(self, dim1, dim2):
        W = np.random.randn(dim1, dim2) * np.sqrt(2. / dim1)  
        b = np.zeros(shape=(1,dim2))  
        return W, b


"""                
class Adam(Optimizer):
    
    def __init__(self, lr):
        self.lr = lr
    
    def step(self, layers):
        for layer in layers:
            if isinstance(layer, Layer):
                
                  
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db
                
                
class Momentum(Optimizer):
    
    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = 0
        
    def step(self, layers):
        for layer in layers:
            if isinstance(layer, Layer):
                
                
                self.v = self.beta * self.v + (1-self.beta)*layer.dW
                                          
                layer.W -= self.lr * self.v
                layer.b -= self.lr * layer.db

class RmsProp(Optimizer):
    
    def __init__(self, lr):
        self.lr = lr
    
    def step(self, layers):
        for layer in layers:
            if isinstance(layer, Layer):  
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db
                
class SGD(Optimizer):
    
    def __init__(self, lr):
        self.lr = lr
    
    def step(self, layers):
        for layer in layers:
            if isinstance(layer, Layer):  
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db
"""