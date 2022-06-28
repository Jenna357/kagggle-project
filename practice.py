# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

# 데이터 파일 읽기 예제
import pandas as pd
import numpy as np
X_test = pd.read_csv("data/X_test.csv")
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

# 사용자 코딩
X_test['gubun'] = 'test'
X_train['gubun'] = 'train'
X_total = pd.concat([X_test, X_train])
X_total.head()

df_y = y_train.drop('cust_id', axis = 1)
df_y = df_y['gender']
df_x = X_total.loc[:,['환불금액', '총구매액', '최대구매액','주구매상품','내점일수','주말방문비율','내점당구매건수','구매주기','gubun']]

df_x['환불금액'] = df_x['환불금액'].fillna(0)
df_x['주구매상품'] = df_x['주구매상품'].replace({'남성 캐주얼': '남성의류', '남성정장' : '남성의류','남성 트랜디':'남성의류', '디자이너': '의류', '시티웨어':'의류', '란제리/내의':'의류', '캐주얼':'의류', '셔츠':'의류', '농산물':'식품', '수산품':'식품', '육류':'식품', '축산가공':'식품', '젓갈/반찬':'식품', '가공식품':'식품','건강식품':'식품','피혁잡화': '잡화', '일용잡화':'잡화', '생활잡화':'잡화', '섬유잡화':'잡화','대형가전':'가전','소형가전':'가전', '식기':'주방용품', '보석':'액세서리', '구두':'잡화', '골프':'스포츠','모피/피혁':'의류', '트래디셔널':'의류', '커리어':'의류', '악기':'기타', '차/커피':'식품', '주방가전':'가전', '주류':'식품'})

# 파생변수
#fm = df_y[(df_y == 0)].index
#ma = df_y[(df_y == 1)].index

#print(df_x.loc[fm, '환불금액'].mean())   # 여자의 환불금액이 큼
#print(df_x.loc[ma, '환불금액'].mean())

#print(df_x.loc[fm, '총구매액'].mean())   # 여자의 총구매액이 큼
#print(df_x.loc[ma, '총구매액'].mean())

#print(df_x.loc[fm, '최대구매액'].mean())   # 여자의 최대구매액이 큼
#print(df_x.loc[ma, '최대구매액'].mean())

#print(df_x.loc[fm, '주말방문비율'].mean())   # 남자의 주말방문비율이 큼
#print(df_x.loc[ma, '주말방문비율'].mean())

#print(df_x.loc[fm, '내점당구매건수'].mean())   # 여자의 내점당구매건수가 큼
#print(df_x.loc[ma, '내점당구매건수'].mean())

#print(df_x.loc[fm, '구매주기'].mean())
#print(df_x.loc[ma, '구매주기'].mean())  # 남자의 구매주기가 큼

#print(df_x.loc[fm, '내점일수'].mean())  # 여자의 내점일수가 큼
#print(df_x.loc[ma, '내점일수'].mean())  

df_x['환불비율'] = (df_x['총구매액'] + df_x['환불금액']) / (df_x['총구매액'] + 1)
df_x['총구매액2'] = (df_x['총구매액'] + df_x['환불금액'])
df_x['최대비율'] = df_x['최대구매액'] / (df_x['총구매액2']+1)
#df_x['방문비율'] = df_x['주말방문비율'] / (df_x['총구매액2']+1) 
df_x['방문주기'] = (df_x['주말방문비율']+1) * (df_x['구매주기']+1)
#df_x['평균금액'] = df_x['총구매액2'] / df_x['내점일수']

df_x = df_x.drop(['총구매액'], axis = 1)

from sklearn.preprocessing import LabelEncoder
df_x['주구매상품'] = LabelEncoder().fit_transform(df_x['주구매상품'])

df_x_tr = df_x.loc[df_x['gubun']=='train',:]
df_x_tr = df_x_tr.drop('gubun', axis= 1)

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df_x_tr, df_y, random_state = 0)

#from sklearn.ensemble import RandomForestClassifier as rf_c
#m_rf = rf_c(random_state = 0)
#m_rf.fit(train_x, train_y)
#print(m_rf.score(train_x, train_y))  # 1 
#print(m_rf.score(test_x, test_y))    # 62.74

from sklearn.ensemble import GradientBoostingClassifier as gb_c
m_gb = gb_c(random_state = 0, learning_rate = 0.02)
m_gb.fit(train_x, train_y)
print(m_gb.score(train_x, train_y))  # 74.4   -> 67.65
print(m_gb.score(test_x, test_y))    # 64.57  -> 65.37
#imp1 = pd.DataFrame(m_gb.feature_importances_, train_x.columns)
#print(imp1.sort_values(0, ascending= False))

df_x_te = df_x.loc[df_x['gubun']=='test',:]
df_x_te = df_x_te.drop('gubun', axis= 1)

pred = m_gb.predict_proba(df_x_te)[:,1]

#score_tr = [] ; score_te =[]
#v_lr = np.arange(0.01, 0.5, 0.01)

#for i in v_lr:
#	m_gb = gb_c(random_state = 0, learning_rate=i)
#	m_gb.fit(train_x, train_y)
#	score_tr.append(m_gb.score(train_x, train_y))
#	score_te.append(m_gb.score(test_x, test_y))
	
#tun1 = pd.DataFrame({'tun':v_lr, 'train' : score_tr, 'test': score_te})
#print(tun1.sort_values('test', ascending=False))

result = pd.DataFrame({'custid': X_test['cust_id'], 'gender':pred})
result.to_csv('2011110825.csv', index = False)

# 답안 제출 참고
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# pd.DataFrame({'cust_id': X_test.cust_id, 'gender': pred}).to_csv('003000000.csv', index=False)

