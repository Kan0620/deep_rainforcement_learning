#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:16:44 2021

@author: nakaharakan
"""

import numpy as np
import random


'''
class environment()で環境クラスを定義する 

.reset()で初期化してその状態を返す


.step(action=(行動のindex))で
next_s(その行動をすると得られる次のs,ニューラルネットに入れる状態にしておく)、
reward、done(エピソードが終わったかの真理値終わったらTrue)を返す 
=================================================================

関数　create_model()で上のnext_sを入力、行動価値を出力にもつmodelを返す

Kerasで実装

==================================================================

class DDQN_agent()でエージェントを定義する。引数はmodel、環境（上で定義したようなクラスの
インスタンス）、actionの数n_action、学習率alpha、割引率g、モデルを保存するときの名前save_nameと
下で説明するn_count,n_test,finish_score,using_data_rateの10個

.new_epsilon(epoch)でそのepochでのepsilonを返す

.create_train_data(epsilon,count)でn_count回エピソードが終わるまでepsilon-greedyで探索
sのarray,aのindexのarray ,rのarray,s'のarrayをを作りDDQNのアルゴリズムにしたがって
教師信号の入力xと出力yを返す。この時データの時系列を破壊するため得られた履歴のusing_data_rate*100%だけ使用


.test()で今のエージェントが100%自分で行動選択した場合n_test回のエピソードで
一回あたり平均どれだけの利得を得られたかを返す

.save()でモデルを保存 モデル名は引数からのsave_name

.fit()でepochと.test()の値を表示しながら学習save()も毎epochする



'''



class DDQN_agent():
    
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

from keras.layers import Dense,Input
from keras import Model


        
def model():
    
    inputs=Input(shape=(4))
        
    x=Dense(32,activation='relu')(inputs)
        
    x=Dense(32,activation='relu')(x)
        
    y=Dense(2,activation='linear')(x)   
    
    model=Model(inputs=inputs,outputs=y)
        
    model.compile(
                loss='mse',
                optimizer='adam',
                )
    
    return model

    
def main():
    
    env=environment()
    
    agent=DDQN_agent(model=model(),
                    env=env,
                    n_action=2,
                    alpha=0.5,
                    g=0.9,
                    n_count=5,
                    using_data_rate=0.5,
                    n_test=5,
                    finish_score=200,
                    save_name='DDQN')
    
    agent.fit()

            
if __name__=='__main__':
    
    main()

           