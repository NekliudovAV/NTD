import ast
import operator as op
import pandas as pd
import numpy as np
from seuif97 import *
from scipy.interpolate import interp1d, LinearNDInterpolator # импортируем методы интерполяции
from math import sqrt, sin, log 
import re
# Находим все идентификаторы с точками


# Расчёт вспомогательных функций
def calc_Pw(T):
        # Saturation pressure at a given temperature
        return pd.Series([tx2p(T[i],0)/ 0.0980665 for i in T.index],T.index)
    
def calc_Hw(T):
        # Enthalpy of water
        return pd.Series([tx2h(T[i],0) for i in T.index],T.index)/4.186
def calc_Hs(T):
        # Enthalpy of steam at the saturation point
        return pd.Series([tx2h(T[i],1) for i in T.index],T.index)/4.186

def calc_T(P):
        # Steam temperature at the saturation point at a given pressure
        return pd.Series([px2t((P[i]+1)* 0.0980665,1) for i in P.index],P.index)

def calc_H(P,T):
        Hs=pd.Series([pt2h((P[i]+1)* 0.0980665,T[i])  for i in T.index],T.index)
        Hs=Hs/4.186
        Hs_=pd.Series([tx2h(T[i],1) for i in T.index],T.index)
        Hs_=Hs_/4.186
        Hs[Hs_>Hs]=Hs_[Hs_>Hs]
        return Hs

def clip(df,min_,max_):
    return df.clip(min_,max_)


def add_curve(Curves,Name,X,F):
        n=np.shape(X);
        if len(n)==1: # Интерполяция одномерных функций
            Curves.update({Name:interp1d(X,F,bounds_error=False, fill_value='extrapolate')})
        else:         # Интерполяция многомерных функций
            Curves.update({Name:LinearNDInterpolator(X, F,rescale=True)})
        return Curves

def preprocess_dotted_variables(code):    
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+)\b'
    def replacer(match):
        return match.group(1).replace('.', '___')
    return re.sub(pattern, replacer, code)

def rereplace(Name):
    return Name.replace('___','.')
    


class ExpressionEvaluator:
    def __init__(self, df):
        df.columns=[i.replace('.','___') for i in df.keys() ]
        self.calculated=[]
        self.df = df
        self.ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,

            # Операции сравнения
            ast.Eq: op.eq, ast.NotEq: op.ne,
            ast.Lt: op.lt, ast.LtE: op.le,
            ast.Gt: op.gt, ast.GtE: op.ge,
            
            # Логические операции
            ast.And: lambda x, y: x & y,  # Для pandas Series
            ast.Or: lambda x, y: x | y,   # Для pandas Series            
        }
        
        self.functions = {
            'fig':self.calc_curve,
            'clip':clip,
            'tw2p':calc_Pw,
            'tw2h':calc_Hw,
            'ts2h':calc_Hs,
            'px2t':calc_T,
            'pt2h':calc_H,
            'sqrt': np.sqrt,
            'sin': np.sin,
            'log': np.log,
            'sum': np.sum,
            # Добавьте другие функции по необходимости
        }
        self.curvs={}
        
    def get_df(self):
        df=self.df.copy()
        df.columns=[i.replace('___','.') for i in df.keys() ]
        return df
        
    def calc_curve(self,Name,*X):
        #print('calc_curve')
        #print('Name:',Name)
        #print('X:',*X)
        return self.curvs[Name](*X)
        
    def add_curve(self,Name,X,F):
        n=np.shape(X);
        if len(n)==1: # Интерполяция одномерных функций
            self.curvs.update({Name:interp1d(X,F,bounds_error=False, fill_value='extrapolate')})
        else:         # Интерполяция многомерных функций
            self.curvs.update({Name:LinearNDInterpolator(X, F,rescale=True)})
        return self.curvs
    

    def eval_expr(self, expr):
        node = ast.parse(expr, mode='eval')
        return self._eval(node.body)

    def _eval(self, node):
        if isinstance(node, ast.Num):  # Число
            return node.n
        elif isinstance(node, ast.Str):  # Строковый литерал
            return node.s    
        elif isinstance(node, ast.Name):  # Столбец DataFrame
            return self.df[node.id]
        elif isinstance(node, ast.BinOp):  # Бинарная операция (+, -, *, /)
            left = self._eval(node.left)
            right = self._eval(node.right)
            return self.ops[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):  # Унарная операция (например, -x)
            return self.ops[type(node.op)](self._eval(node.operand))
        elif isinstance(node, ast.Call):  # Функции (sqrt(), sin() и т.д.)
            func_name = node.func.id
            args = [self._eval(arg) for arg in node.args]
            return self.functions[func_name](*args)
        elif isinstance(node, ast.Compare):  # Обработка сравнений
            left = self._eval(node.left)
            # Обрабатываем цепочку сравнений (например: 30 <= age <= 40)
            result = left
            for operation, comparator in zip(node.ops, node.comparators):
                right = self._eval(comparator)
                result = self.ops[type(operation)](result, right)
            return result        
        else:
            raise ValueError(f"Неподдерживаемая операция: {type(node).__name__}")

    def calc_expr(self,expr,new_column_name):
        # Вычисляем выражение
        try:
            result = self.eval_expr(expr)
            self.calculated.append(new_column_name)
            print(f"Выражение:{new_column_name} = {expr} OK!\n") #Результат: {result}\n
        except Exception as e:
            print(f"Ошибка в выражении '{new_column_name}={expr}': {str(e)}")
            result = df.eval(expression)
        
        # Если результат - Series (один столбец), добавляем в DataFrame
        if isinstance(result, (pd.Series, np.ndarray)):
            
            kwargs = {new_column_name: result}
            #self.df[new_column_name] = #result.values
            self.df = self.df.assign(**kwargs)
        else:
            # Если результат скалярный, применяем ко всем строкам
            self.df[new_column_name] = result
        return self.df    
        
    def calc_expressions(self,expressions):
        # Вычисление выражений
        for expr, col_name in expressions:
            print(col_name,'=',expr)
            self.calc_expr(expr, col_name)
        return  self.df 
        
    def calc_expressions_eq(self,expressions):
        # Вычисление выражений
        for expression in expressions:
            expression=preprocess_dotted_variables(expression)
            col_name, expr = expression.split('=')
            #print(col_name,'=',expr)
            self.calc_expr(expr, col_name)
        return  self.df 
    
    def get_calc(self):
        return self.df[self.calculated]


