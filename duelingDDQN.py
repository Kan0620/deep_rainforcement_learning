#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 02:21:30 2021

@author: nakaharakan
"""

import numpy as np
import random

class duelingDDQN_agent():
    
    def __init__(self,model,env,n_action,alpha,g,n_count,using_data_rate,n_test,finish_score,save_name):
        
        model.summary()
        
        #２つのモデルを定義
        
        self.main_model=model
        
        self.target_model=model
        
        self.env=env
        
        self.n_action=n_action
        
        self.alpha=alpha
        
        self.g=g
        
        self.n_count=n_count
        
        self.using_data_rate=using_data_rate
        
        self.n_test=n_test
        
        self.finish_score=finish_score
        
        self.save_name=save_name
        
        
        
    def new_epsilon(self,epoch): 
        
        #指数的に減衰させる
        
        return 0.95**(epoch-1)
    
    def create_train_data(self,n_count,using_data_rate,n_action,epsilon):
        
        s=[]
        a_index=[]
        r=[]
        next_s=[]
        
        #n_count回分のデータ作成
        
        for i in range(n_count):
            
            #エピソードが終わっているかの変数doneをFalseに初期化、環境も初期化
            
            done=False
            
            observation=self.env.reset()
            
            while not done:                
                
                if random.random()<using_data_rate:#今からの処理で得られるデータが学習に使われる場合
                    
                
                    
                    s.append(observation)
                
                    if random.random()<epsilon:#探索
                        
                            action=np.random.choice(n_action)
                        
                    else:#活用
                         
                            action=np.argmax(self.main_model.predict(np.array([observation]))[0])
                        
                    a_index.append(action)
                        
                    observation,reward,done=self.env.step(action)
                
                    r.append(reward)
                
                    next_s.append(observation)
                
        
                else:#使われない場合
                    if random.random()<epsilon:#探索
                        
                            action=np.random.choice(n_action)
                        
                    else:#活用
                         
                            action=np.argmax(self.main_model.predict(np.array([observation]))[0])
                    
                    observation,reward,done=self.env.step(action)
                    
        
        s=np.array(s)
        a_index=np.array(a_index)
        r=np.array(r)
        next_s=np.array(next_s)
        
        #x、yはニューラルネットの入力と出力
        x=s
        
        y=self.main_model.predict(s)
        
        #Double Q-learningの更新式でyを更新
        
        
        y[np.array([i for i in range(len(x))]),a_index]=\
        (1-self.alpha)*y[np.array([i for i in range(len(x))]),a_index]+\
        self.alpha*(r+self.g*self.target_model.predict(next_s)\
                    [np.array([i for i in range(len(x))]),y.argmax(axis=1)])
        
        '''
        ↓上のコードの意味↓　行が対応しています y＝main_model(s)
        
        
        新しいQ値=
        (1-α)main_Q(s,a)+
        α(r+γ*target_Q(s,a'))
        ただしa'はmain_modelの出力値のargmax
        '''        
                
        return x,y        
    
    
                
    def test(self,n_test):
        
        r=0
        
        for i in range(n_test):
            
            done=False
            
            observation=self.env.reset()
            
            while not done:
                
                                
                action=np.argmax(self.main_model.predict(np.array([observation]))[0])
                
                observation,reward,done=self.env.step(action)
                
                #r+=reward
            r+=self.env.count   
                
        return r/n_test
    
    
    
    def save(self,save_name):
        
        self.main_model.save(save_name+'.hdf5')
        
        
    
    def fit(self):
        
        epoch=1
        
        score=0
        
        while  self.finish_score>score: #学習の継続条件
            
            epsilon=self.new_epsilon(epoch)#新しいepsilon取得
            
            #教師データ取得
            x,y=self.create_train_data(self.n_count,self.using_data_rate,self.n_action,epsilon)
            
            #main_modelの重みをtarget_modelの重みに共有
            
            self.target_model.set_weights(self.main_model.get_weights())
            
            #main_model学習
            self.main_model.fit(x,y,
                       epochs=3,
                       verbose=0
                       )
            #モデル保存
            self.save(self.save_name)
            
            #現在のモデルの平均スコア取得
            score=self.test(self.n_test)
            
            print('epoch:'+str(epoch)+'  score:'+str(score))
            
            epoch+=1
            
        print('fitting finished')
        
 














       
################ cart_poleでの使用例 ##############
        
import gym

class environment():
    
    def __init__(self):
        
        self.env=gym.make('CartPole-v0')
        
        self.env.reset()
        
        self.count=0
        
    def reset(self):
        
        self.count=0
        
        return self.env.reset()
    
    def step(self,action_index):
        
        observation,reward,done,info=self.env.step(action_index)
        
        self.count+=1
        
        if reward==1:
            
            reward=0
            
        if self.count==200:
            
            reward=1
            
        if done:
            
            reward=-1
            
        return observation,reward,done
    

from keras.layers import Dense,Input,concatenate,Lambda
import keras.backend as K
from keras import Model
from tensorflow import stop_gradient

        
def model():
    
    inputs=Input(shape=(4))
        
    x=Dense(32,activation='relu')(inputs)
        
    v=Dense(32,activation='relu')(x)
    
    v=Dense(1,activation='linear')(v) #V(s)
    
    a=Dense(32,activation='relu')(x)
    
    a=Dense(2,activation='linear')(a) #A(a|s)
    
    y=concatenate([v,a])
    
    y=Lambda(lambda x:K.expand_dims(x[:,0],-1)+x[:,1:]-stop_gradient(K.mean(x[:,1:],keepdims=True)),\
             output_shape=(2))(y)
    
    '''
    
    ↓上のコードの意味↓　
    
    K.expand_dims(x[:,0],-1)　→ x[:,0]はconcatenate層の出力のindex0の要素であるV(s)を表していて、
    各A(a|s)と足せるようにactionの数だけexpandしている
    
    x[:,1:]はconcatenate層の出力のindex1以降の要素で各A(a|s)のこと
    
    stop_gradient(K.mean(x[:,1:],keepdims=True)) → 各A(a|s)の平均、定数扱いなので勾配計算に
    関わらないようにstop_gradientしている
    
    '''
    
    model=Model(inputs=inputs,outputs=y)
        
    model.compile(
                loss='mse',
                optimizer='adam',
                )
    
    return model

    
def main():
    
    env=environment()
    
    agent=duelingDDQN_agent(model=model(),
                    env=env,
                    n_action=2,
                    alpha=0.5,
                    g=0.9,
                    n_count=25,
                    using_data_rate=0.7,
                    n_test=5,
                    finish_score=200,
                    save_name='DDQN')
    
    agent.fit()

            
if __name__=='__main__':
    
    main()