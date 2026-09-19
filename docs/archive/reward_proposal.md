# BlackOut 확률 기반 보상 설계안

> **보관 문서.** Unity 쪽 옛 보상(팀 Ψ + 개인 Φ)의 원래 설계안이다. Run 11은 Python 리워드 v2(`docs/reward_v2_design.md`)를 쓴다.
> 현재 Unity 구현은 `../blackout/Documentation/reward_shaping.md`. 여기 링크된 `examples/evaluate_reward_model.py`,
> `evaluate_reward_alignment.py`는 삭제됐다(`git log --diff-filter=D`로 찾을 수 있음).

## 1. 목적과 설계 원칙

BlackOut의 최종 목적은 개별 아이템 획득, 킬 또는 직업 변환 횟수를 늘리는 것이 아니라 상대 팀보다 높은 최종 점수를 만들어 승리하는 것이다. 따라서 중간 사건에 고정된 임의 보상을 직접 부여하지 않고, 그 사건이 다음 두 확률을 얼마나 변화시켰는지를 보상으로 사용한다.

1. 각 배터리가 다음 흡수 시점에 어느 팀 점수로 확정될 확률
2. 현재 상태에서 각 팀이 최종적으로 승리할 확률

전투, 사망, 약탈, 운반, 창고 방어 및 직업 변환은 모두 위 확률을 바꾸는 수단으로 평가한다.

설계 목표는 다음과 같다.

- 최종 승패가 최상위 목적이어야 한다.
- 양 팀의 효용은 가능한 한 zero-sum이어야 한다.
- 배터리는 아이템 개수가 아니라 실제 수량에 비례해 평가되어야 한다.
- 흡수 전 창고 점수는 약탈 위험을 반영한 조건부 기댓값으로 평가되어야 한다.
- 아이템을 반복해서 줍고 놓거나 창고에서 꺼냈다 넣는 순환으로 양의 보상을 축적할 수 없어야 한다.
- 킬과 사망은 고정 보상이 아니라 그 결과로 바뀐 전략적 상태의 가치로 평가해야 한다.
- Collector에서 Hunter 또는 Carrier로 바뀌는 semi-irreversible 직업 선택은 미래 선택권의 상실까지 포함해 평가해야 한다.
- 에이전트별 보상의 팀 내 합은 의도한 팀 보상과 정확히 같아야 한다.

---

## 2. 환경에서 사용하는 기호

팀 (k\in\{A,B\}), 상대 팀을 (-k), 팀당 에이전트 수를 (N=5)라고 한다.

| 기호 | 의미 |
|---|---|
| (s_t) | 시점 (t)의 전역 Markov 상태 |
| (q_b) | 배터리 (b)의 실제 수량 |
| (L_k) | 팀 (k)의 이미 흡수된 확정 점수 |
| (T_{\mathrm{abs}}) | 다음 흡수 시점 |
| (T) | 에피소드 종료 시점 |
| (Z_k) | 최종 결과 효용: 승리 (+1), 무승부 (0), 패배 (-1) |
| \(\gamma\) | 한 환경 transition의 할인율 |
| \(\eta\) | 확률 기반 shaping의 전체 크기 |

현재 환경은 0.02초마다 한 transition을 생성하므로, 장기 결과를 보존하기 위한 초기 권장값은 다음과 같다.

\[
\boxed{\gamma=0.99995,\qquad\eta=0.25}
\]

이때 20초 뒤 보상의 할인 가중치는 약 (0.951), 420초 뒤 보상의 가중치는 약 (0.350)이다.

---

## 3. 배터리별 다음 흡수 결과의 확률 모델

### 3.1 범주형 결과

현재 맵에 존재하는 각 비확정 배터리 (b)에 대해 팀 (k) 관점의 다음 흡수 결과를 다음 확률변수로 둔다.

\[
X_b\in\{+1,-1,0\}
\]

- (X_b=+1): 다음 흡수 시 팀 (k)의 점수로 확정
- (X_b=-1): 다음 흡수 시 상대 팀의 점수로 확정
- (X_b=0): 다음 흡수 시 어느 팀에도 확정되지 않음. 필드, 운반 중, 파괴 상태 등을 포함한다.

각 조건부 확률을 다음과 같이 정의한다.

\[
\pi_b^k=P(X_b=+1\mid s)
\]

\[
\pi_b^{-k}=P(X_b=-1\mid s)
\]

\[
\pi_b^0=1-\pi_b^k-\pi_b^{-k}
\]

배터리 하나의 다음 흡수 시 기대 점수차 기여는 다음과 같다.

\[
\mathbb E[q_bX_b\mid s]
=q_b(\pi_b^k-\pi_b^{-k})
\]

이 정의는 배터리가 필드, 아군 운반, 적군 운반, 아군 창고 또는 적군 창고 중 어디에 있든 동일하게 적용된다.

### 3.2 예측 모델

배터리별 확률은 다음과 같은 3분류 모델로 예측한다.

\[
(\pi_b^A,\pi_b^B,\pi_b^0)
=\operatorname{softmax}(f_\theta(x_b,s))
\]

최소 입력 feature는 다음과 같다.

- 배터리 수량, 위치, 소유 및 운반 상태
- 보호 창고 여부와 다음 흡수까지 남은 시간
- 배터리 또는 운반자에서 양 팀 창고까지의 최단 경로 시간
- 배터리 또는 운반자에 도달할 수 있는 양 팀 Collector/Carrier/Hunter의 최단 경로 시간
- 운반자를 죽일 수 있는 적의 수와 호위 가능한 아군의 수
- 클래스 상성, 실제 이동 속도, 충돌 크기 및 활성 버프/디버프
- 창고 주변의 국소 수적 우세와 진입 가능한 경로 수
- 팀별 Collector/Hunter/Carrier 수와 Carrier 슬롯 사용 여부
- 확정 점수, 목표 점수까지 남은 양 및 에피소드 잔여 시간

거리는 직선거리가 아니라 벽과 팀별 진입 제한을 반영한 최단 경로 시간으로 계산해야 한다.

### 3.3 초기 hazard 근사

완전한 3분류 모델을 만들기 전에는 창고 내 배터리의 탈취 위험률을 이용해 근사할 수 있다. 흡수까지 남은 시간이 \(\tau\)이고 순간 탈취 위험률이 \(\lambda_b^{\mathrm{steal}}(u\mid s)\)이면:

\[
P_b^{\mathrm{survive}}
=\exp\left(-\int_0^\tau\lambda_b^{\mathrm{steal}}(u\mid s)\,du\right)
\]

위험률을 구간 내 상수로 근사하면:

\[
P_b^{\mathrm{survive}}=e^{-\lambda_b\tau}
\]

보호 창고에는 \(\lambda_b=0\)을 적용한다. 외부 창고의 \(\lambda_b\)에는 적 수집 유닛의 도달 시간, 아군 Hunter의 방어 가능성, 활성 속도 효과 및 출구 차단 정도를 반영한다.

hazard 모델은 “아군이 지킬 확률”만 표현하므로, 탈취 후 상대 팀이 실제로 확정할 확률까지 표현하려면 최종적으로 3분류 모델로 전환해야 한다.

---

## 4. 다음 흡수 후의 확률적 점수 우세

### 4.1 기대 점수차

팀 (k) 관점의 다음 흡수 후 점수차 기댓값을 다음처럼 둔다.

\[
\boxed{
\mu_k^{\mathrm{abs}}(s)
=L_k-L_{-k}
+\sum_bq_b(\pi_b^k-\pi_b^{-k})
}
\]

예를 들어 수량 8 배터리에 대해:

\[
(\pi_b^k,\pi_b^{-k},\pi_b^0)=(0.65,0.15,0.20)
\]

이면 명목상 8점이지만 기대 점수차 기여는 다음과 같다.

\[
8(0.65-0.15)=4
\]

### 4.2 점수차 분산

배터리별 귀속을 조건부 독립으로 근사하면:

\[
\operatorname{Var}(X_b\mid s)
=\pi_b^k+\pi_b^{-k}-(\pi_b^k-\pi_b^{-k})^2
\]

따라서 다음 흡수 후 점수차 분산은:

\[
\boxed{
(\sigma_k^{\mathrm{abs}})^2
=\sum_bq_b^2
\left[
\pi_b^k+\pi_b^{-k}-(\pi_b^k-\pi_b^{-k})^2
\right]
+\sigma_{\mathrm{model}}^2
}
\]

이다. \(\sigma_{\mathrm{model}}^2\)는 독립성 근사와 예측 모델의 잔여 오차를 보상한다. 여러 배터리의 결과가 같은 운반자나 같은 전투에 강하게 묶이는 경우에는 Monte Carlo joint rollout로 공분산까지 추정하는 편이 더 정확하다.

### 4.3 다음 흡수 후 앞설 확률

점수차를 정규분포로 근사하면:

\[
Y_k^{\mathrm{abs}}\sim
\mathcal N\left(\mu_k^{\mathrm{abs}},(\sigma_k^{\mathrm{abs}})^2\right)
\]

다음 흡수 우세 효용은 다음과 같다.

\[
\boxed{
U_k^{\mathrm{abs}}(s)
=P(Y_k^{\mathrm{abs}}>0\mid s)-P(Y_k^{\mathrm{abs}}<0\mid s)
=2\Phi\left(
\frac{\mu_k^{\mathrm{abs}}}
{\sqrt{(\sigma_k^{\mathrm{abs}})^2+\epsilon}}
\right)-1
}
\]

따라서 평균 점수차가 같아도 결과 불확실성이 낮은 상태를 더 우세하게 평가한다.

---

## 5. 최종 승리 확률

다음 흡수 우세가 최종 승리를 완전히 대변하지는 않으므로 별도의 calibrated outcome model을 둔다.

\[
(p_k^W,p_k^D,p_k^L)=g_\omega(s)
\]

최종 승리 효용은:

\[
\boxed{
U_k^{\mathrm{win}}(s)=p_k^W-p_k^L
}
\]

로 정의한다. 모델 입력에는 다음 흡수 귀속 확률 외에도 남은 배터리, 남은 흡수 횟수, 직업 구성, Carrier 슬롯, 활성 효과, 위치 장악도, 목표 점수까지의 거리 및 잔여 시간을 포함한다.

배터리 모델과 승률 모델은 self-play rollout으로 학습하고 Brier score, negative log-likelihood, reliability diagram 및 expected calibration error로 교정한다. 확률값은 분류 정확도보다 calibration이 중요하다.

---

## 6. 상태 potential과 팀 보상

팀 (k)의 bounded state potential은 다음과 같다.

\[
\boxed{
\Psi_k(s)
=0.6U_k^{\mathrm{win}}(s)
+0.4U_k^{\mathrm{abs}}(s)
}
\]

반대 팀은 같은 전역 예측을 반대 관점에서 사용한다.

\[
\Psi_{-k}(s)=-\Psi_k(s)
\]

팀 보상은:

\[
\boxed{
R_{k,t}
=z_{k,t}
+\eta[\gamma\Psi_k(s_{t+1})-\Psi_k(s_t)]
}
\]

으로 계산한다.

\[
z_{k,t}=
\begin{cases}
+1,&t=T,\ k\text{ 승리}\\
0,&t=T,\text{ 무승부 또는 }t<T\\
-1,&t=T,\ k\text{ 패배}
\end{cases}
\]

terminal transition에서는 다음 terminal state의 potential을 0으로 설정한다. 예측 모델과 potential 함수가 한 정책 학습 구간 동안 고정되어 있으면 shaping 항은 discounted return에서 telescoping되어 최종 승패 목적의 최적 정책을 바꾸지 않는다.

---

## 7. 에이전트별 크레딧 할당

모든 팀원에게 단순히 \(R_k/5\)를 줄 수 있지만 누가 상태 개선을 일으켰는지를 표현하지 못한다. 에이전트별 반사실 다음 상태를 이용한다.

- 실제 다음 상태: \(s_{t+1}\)
- 에이전트 (i)의 효과만 제거한 다음 상태: \(s_{t+1}^{(-i)}\)

개별 marginal은:

\[
d_i
=\gamma\left[
\Psi_k(s_{t+1})-\Psi_k(s_{t+1}^{(-i)})
\right]
\]

으로 정의한다. 합의 중복 또는 누락을 잔차 배분으로 교정한다.

\[
\boxed{
r_{i,t}
=\eta d_i
+\frac{R_{k,t}-\eta\sum_{j\in k}d_j}{5}
}
\]

따라서 항상:

\[
\sum_{i\in k}r_{i,t}=R_{k,t}
\]

이다.

반사실 상태는 다음 방식으로 구성한다.

- 이동: 에이전트 (i)만 이전 위치에 둔다.
- 획득/절도: (i)가 해당 아이템을 획득하지 않은 상태로 둔다.
- 적재: (i)가 운반 상태를 유지한 것으로 둔다.
- 전투: (i)가 전투에 참가하지 않은 결과를 사용한다.
- 직업 변환: (i)만 Collector로 유지한다.

현재 QMIX 학습기는 팀원 보상을 합산한 값만 replay buffer에 저장하므로 위 개별 차이를 실제 학습에 사용하려면 에이전트별 reward vector와 작은 individual auxiliary TD loss를 추가해야 한다. 그렇지 않아도 합의 5배 증폭을 방지하고 디버깅 가능한 크레딧을 남긴다는 의미는 있다.

---

## 8. 직업 변환 모델

### 8.1 semi-irreversible 선택

Collector는 Hunter 또는 Carrier로 변환할 수 있지만 임의로 Collector로 돌아올 수 없다. 사망하면 본진에서 Collector로 부활하므로 완전히 비가역적이지는 않으나 다음 비용을 갖는다.

- 기존 Collector의 수집 및 대Carrier 능력 상실
- 다른 직업으로 변환할 미래 선택권 상실
- 원상 복귀를 위해 사망해야 함
- 사망 위치에서 본진으로 강제 이동
- 보유 아이템 파괴
- Carrier는 팀 전체에서 유일한 슬롯을 점유

따라서 직업별 고정 변환 보상은 사용하지 않는다.

### 8.2 반사실 직업 가치

에이전트 (i)가 직업 (c)로 변환하는 가치는 다음과 같다.

\[
\boxed{
\Delta_i^{\mathrm{role}}
=\mathbb E[Z_k\mid\operatorname{do}(c_i=c),s]
-\mathbb E[Z_k\mid\operatorname{do}(c_i=\mathrm{Collector}),s]
}
\]

직업이 승률 및 배터리 귀속 예측 모델의 상태에 포함되면 별도 `transformReward` 없이 \(\Psi(s_{\mathrm{after}})-\Psi(s_{\mathrm{before}})\)에 자동으로 반영된다.

### 8.3 Collector의 선택권 가치

Collector의 가치는 현재 수집 능력뿐 아니라 미래 변환 선택권을 포함한다.

\[
V_i^{\mathrm{Collector}}
=V_i^{\mathrm{collect}}+O_i^{\mathrm{transform}}
\]

\[
O_i^{\mathrm{transform}}
=\max\left(0,V_i^H-C_i^H,V_i^C-C_i^C\right)
\]

- \(V_i^H,V_i^C\): 지금 Hunter 또는 Carrier가 되었을 때의 조건부 가치
- \(C_i^H,C_i^C\): 성소까지의 시간, 수집 기회비용, 직업 고착 및 복귀 비용

변환 모델이 이 option value를 포함하지 않으면 모든 Collector가 당장의 속도 또는 전투 이득만 보고 너무 빨리 변환하는 편향이 생긴다.

### 8.4 Carrier 슬롯의 기회비용

Carrier는 팀당 한 명만 존재하므로 에이전트 (i)의 Carrier 전환 가치는 팀 내 다른 에이전트가 그 슬롯을 사용할 기회를 잃는 비용을 포함해야 한다.

\[
V_{i,\mathrm{net}}^C
=V_i^C-\max_{j\ne i}V_j^C
\]

실제 모델에서는 “지금 (i)가 Carrier가 된 세계”와 “슬롯을 열어 두어 팀 정책이 이후 사용할 수 있는 세계”의 outcome probability를 비교한다.

### 8.5 Hunter 가치의 조건

Hunter 가치가 커지는 대표 조건은 다음과 같다.

- 적 Carrier 또는 고가 배터리 운반자가 존재
- 아군 외부 창고에 큰 미확정 점수가 있음
- 흡수까지 시간이 길고 방어가 필요함
- 적의 창고 침투 경로를 실제로 차단할 수 있음
- 팀에 Hunter가 부족하고 수집 유닛은 충분함

다음 상황에서는 가치가 작거나 음수가 될 수 있다.

- 남은 배터리는 많지만 아군 Collector가 부족함
- 이미 Hunter가 충분함
- 흡수 직전이라 전장까지 도달할 수 없음
- 점수 열세에서 수집 인력을 추가로 잃음

### 8.6 Carrier 가치의 조건

Carrier 가치가 커지는 대표 조건은 다음과 같다.

- 아직 팀 Carrier가 없음
- 멀리 있는 고가 배터리가 존재
- 적 Hunter가 적거나 안전한 우회 경로가 있음
- 다음 흡수 전에 운반 및 적재가 가능
- 다른 Collector가 충분하여 수집 유연성이 유지됨

다음 상황에서는 가치가 작거나 음수가 될 수 있다.

- 적 Hunter가 운반 경로를 장악
- 남은 배터리가 거의 없음
- 팀의 유일한 Collector를 전환해야 함
- 더 적합한 다른 유닛이 Carrier 슬롯을 사용할 수 있음
- Collector의 대Carrier 전투 능력이 더 중요한 상태

---

## 9. 게임 시나리오별 보상 작동 평가

아래 평가는 확률 모델이 적절히 calibrated되어 있고 한 정책 학습 구간 동안 고정되어 있다는 전제에 기반한다.

### 시나리오 1: 안전한 고가 배터리 수집과 운반

**상황**

- 아군 Collector가 수량 8 배터리를 획득한다.
- 가까운 적 Hunter가 없고 보호 창고까지 안전한 경로가 있다.

**예상 확률 변화**

- \(\pi_b^k\)가 크게 증가한다.
- \(\pi_b^{-k}\)가 감소한다.
- 다음 흡수 우세와 최종 승률이 함께 증가한다.

**예상 보상**

- 팀 보상 양수.
- 실제 획득자에게 가장 큰 positive marginal.
- 경로를 확보한 아군이 있다면 그 에이전트도 이후 transition에서 positive marginal을 받을 수 있다.

**평가**

- 적절하다. 수량 1과 8을 고정값으로 취급하지 않고 실제 기대 기여로 구분한다.

### 시나리오 2: 고가 배터리를 주웠지만 즉시 포위됨

**상황**

- 수량 8 배터리를 획득했지만 적 Hunter가 매우 가까워 탈출 가능성이 낮다.

**예상 확률 변화**

- 획득 이벤트 자체는 발생했지만 \(\pi_b^k\)가 크게 증가하지 않는다.
- 사망 및 파괴 확률이 높아 \(\pi_b^0\)가 높게 유지된다.

**예상 보상**

- 작은 양수 또는 0에 가까움.
- 이후 안전한 경로를 확보하면 점진적으로 양수.
- 사망해 배터리가 파괴되면 앞선 증가분이 되돌아감.

**평가**

- 적절하다. “주웠다”가 아니라 “점수로 만들 가능성이 높아졌다”를 평가한다.

### 시나리오 3: 흡수 직후 외부 창고에 적재

**상황**

- 다음 흡수까지 약 20초가 남았고 외부 창고의 방어가 약하다.

**예상 확률 변화**

- 명목 점수는 즉시 증가하지만 탈취 hazard가 높아 \(\pi_b^k\)는 제한적으로만 증가한다.

**예상 보상**

- 작거나 중간 정도의 양수.
- 보호 창고에 적재했을 때보다 작음.
- 이후 Hunter가 방어 위치를 잡으면 추가 양수.

**평가**

- 적절하다. 명목 점수 증가를 확정 점수처럼 과대평가하지 않는다.

### 시나리오 4: 흡수 직전 외부 창고에 적재

**상황**

- 흡수까지 1초 미만이고 적이 창고에 도달할 수 없다.

**예상 확률 변화**

- \(\pi_b^k\)가 거의 1로 상승한다.
- 점수차 분산은 감소한다.

**예상 보상**

- 분명한 양수.
- 같은 배터리를 흡수 직후 적재하는 경우보다 큼.

**평가**

- 적절하다. 흡수 타이밍을 활용하는 플레이를 유도한다.

### 시나리오 5: 보호 창고에 적재

**상황**

- 운반 거리는 더 길지만 적이 진입할 수 없는 보호 창고에 적재한다.

**예상 확률 변화**

- \(\pi_b^k\)가 거의 1이 된다.
- 귀속 불확실성이 크게 감소한다.

**예상 보상**

- 큰 양수.
- 긴 운반 중 사망 위험은 적재 이전의 확률에 반영되므로 보호성과 거리의 trade-off가 자연스럽게 계산됨.

**평가**

- 적절하다. 단순 최단 거리 창고 선택보다 안전한 고가 배터리 운반을 평가할 수 있다.

### 시나리오 6: 적 창고에서 배터리 절도 성공

**상황**

- 흡수 직전 적 외부 창고에서 수량 8 배터리를 Carrier가 훔친다.
- 탈출 경로도 비교적 안전하다.

**예상 확률 변화**

- 적의 \(\pi_b^{-k}\)가 크게 감소한다.
- 아군의 \(\pi_b^k\)가 증가한다.
- 하나의 배터리가 점수차 양쪽을 동시에 뒤집을 수 있다.

**예상 보상**

- 큰 양수.
- 절도 에이전트에게 큰 positive marginal.
- 진입로를 연 Hunter에게도 실제 반사실 기여가 있다면 양수.

**평가**

- 적절하다. 절도 횟수가 아니라 훔친 수량, 타이밍 및 회수 가능성에 비례한다.

### 시나리오 7: 절도 직후 사망하여 배터리 파괴

**상황**

- 적 창고에서 배터리를 훔쳤지만 즉시 적 Hunter에게 사망한다.

**예상 확률 변화**

- 절도 순간에는 적 확정 확률이 낮아져 양의 변화가 있을 수 있다.
- 사망 후 배터리가 파괴되면 아군 확정 확률도 0이 된다.
- 적의 득점을 막은 가치와 아군 득점 기회를 잃은 가치가 함께 계산된다.

**예상 보상**

- 적 득점만 막아도 유리한 점수 상황이면 순양수 가능.
- 아군이 반드시 그 배터리를 확보해야 역전 가능한 상황이면 작거나 음수 가능.

**평가**

- 상황 의존적으로 적절하다. “훔친 후 죽었으니 항상 실패” 또는 “상대 점수를 없앴으니 항상 성공”으로 고정하지 않는다.

### 시나리오 8: 아이템을 반복해서 줍고 놓음

**상황**

- 같은 배터리를 안전한 위치에서 반복 획득·상실한다.

**예상 확률 변화**

- 동일 상태로 돌아오면 \(\Psi\)도 원래 값으로 돌아온다.

**예상 보상**

- potential 차분의 discounted cycle 합은 거의 0.
- 반복 횟수에 비례한 양의 보상을 축적할 수 없음.

**평가**

- 적절하다. 단, 별도의 고정 pickup 보상을 추가하면 이 성질이 깨진다.

### 시나리오 9: 빈손 적을 안전 지역에서 반복 처치

**상황**

- 적의 배터리 운반이나 창고 공격과 무관한 위치에서 빈손 Collector를 반복 처치한다.

**예상 확률 변화**

- 적은 본진에서 Collector로 부활한다.
- 배터리 귀속 확률과 승률이 거의 변하지 않는다.

**예상 보상**

- 거의 0.

**평가**

- 적절하다. kill farming을 억제한다.
- 반복 처치가 실제로 적의 이동 시간을 지속적으로 손실시킨다면 작은 양수가 생길 수 있으며 이는 실제 전략 가치다.

### 시나리오 10: 배터리를 든 적 Carrier 처치

**상황**

- 적 Carrier가 수량 8 배터리를 들고 적 창고로 접근 중이다.

**예상 확률 변화**

- 적 확정 확률 \(\pi_b^{-k}\)가 크게 감소한다.
- 사망으로 배터리가 파괴되므로 \(\pi_b^0\)가 증가한다.

**예상 보상**

- 방어 팀에 큰 양수.
- 실제 킬러에게 큰 positive marginal.

**평가**

- 적절하다. 동일한 킬이라도 빈손 적보다 훨씬 크게 평가한다.

### 시나리오 11: Hunter끼리 상호 사망

**상황**

- 양 팀 Hunter가 충돌해 동시에 사망한다.

**예상 확률 변화**

- 양측 방어 및 공격 hazard가 동시에 변한다.
- 중요한 창고를 지키던 Hunter와 의미 없는 위치의 Hunter는 가치가 다르다.

**예상 보상**

- 완전히 대칭적인 교환이면 양 팀 모두 0에 가까움.
- 상대 Hunter 제거로 아군 Carrier 경로가 열리면 아군 양수.
- 아군 외부 창고 방어가 무너지면 아군 음수.

**평가**

- 적절하다. combat event 전체를 원자적 transition으로 평가해야 하며 두 사망을 순차적으로 평가하면 순서 의존 오류가 생긴다.

### 시나리오 12: Hunter가 아군 Carrier를 호위

**상황**

- Hunter가 직접 아이템을 들거나 킬하지 않지만 적 접근 경로를 차단한다.

**예상 확률 변화**

- Carrier 생존 확률과 배터리의 아군 확정 확률이 증가한다.

**예상 보상**

- 팀 보상 양수.
- 반사실 상태에서 Hunter를 제거했을 때 확정 확률이 낮아진다면 Hunter에게 positive marginal.

**평가**

- 적절하다. 직접 이벤트가 없는 호위와 지역 통제를 보상할 수 있다는 것이 확률 상태가치 방식의 중요한 장점이다.

### 시나리오 13: 필요할 때 Collector가 Hunter로 변환

**상황**

- 적 Carrier가 고가 배터리를 운반 중이고 아군에는 Hunter가 없다.
- 변환 후 차단 지점까지 제시간에 도착 가능하다.

**예상 확률 변화**

- 적 배터리의 \(\pi_b^{-k}\)가 감소한다.
- 최종 승률이 증가한다.
- Collector 선택권 상실보다 방어 가치가 큼.

**예상 보상**

- 변환 에이전트에게 양수.

**평가**

- 적절하다. `Hunter 변환 = 항상 +x`가 아니라 현재 위협과 도달 가능성에 의해 결정된다.

### 시나리오 14: 너무 많은 Hunter로 조기 변환

**상황**

- 게임 초반 배터리가 많지만 팀 Collector 대부분이 Hunter로 변한다.

**예상 확률 변화**

- 팀의 배터리 수집 및 운반 가능성이 감소한다.
- 남은 Collector의 미래 직업 선택권도 팀 전체에서 부족해진다.

**예상 보상**

- 첫 Hunter는 상황에 따라 양수일 수 있다.
- 추가 Hunter는 marginal이 감소하고 결국 음수가 됨.

**평가**

- 적절하다. outcome model이 직업 구성과 수집 capacity를 충분히 표현해야 한다.

### 시나리오 15: 적절한 Carrier 변환

**상황**

- 멀리 수량 8 배터리가 있고 적 Hunter가 적으며 팀 내 Collector가 충분하다.
- 아직 Carrier 슬롯이 비어 있다.

**예상 확률 변화**

- 해당 배터리와 주변 배터리의 아군 확정 확률이 증가한다.
- 최종 승률이 증가한다.

**예상 보상**

- Carrier 변환 에이전트에게 양수.

**평가**

- 적절하다.

### 시나리오 16: 부적절한 Carrier 슬롯 선점

**상황**

- 적 Hunter 가까이에 있는 유닛이 먼저 Carrier가 된다.
- 다른 Collector는 고가 배터리와 안전 경로에 가깝지만 Carrier 슬롯을 사용할 수 없게 된다.

**예상 확률 변화**

- 현재 Carrier의 생존 확률은 낮다.
- 더 적합한 유닛의 미래 Carrier option이 사라진다.
- 팀의 전체 확정 확률 또는 승률이 감소할 수 있다.

**예상 보상**

- 변환 에이전트에게 음수.

**평가**

- Carrier 슬롯과 Collector option value가 상태에 명시적으로 포함될 때 적절하다. 이를 누락하면 모델은 속도 6이라는 즉시 이득만 보고 잘못된 양의 보상을 줄 수 있다.

### 시나리오 17: 잘못된 직업을 사망으로 리셋

**상황**

- 팀에 Hunter가 지나치게 많고 Collector가 부족하다.
- 한 Hunter가 사망해 본진 Collector로 복귀한다.

**예상 확률 변화**

- 즉시 전투력은 감소하지만 수집 capacity와 미래 변환 option이 회복된다.
- 위치도 본진으로 바뀐다.

**예상 보상**

- 상황에 따라 양수일 수 있다.
- 중요한 창고를 버리고 죽었다면 음수.

**평가**

- 게임 규칙과 정렬되어 있다. 전략적 자살을 허용하고 싶지 않다면 고정 사망 패널티보다 respawn delay 등 게임 규칙을 수정해야 한다.

### 시나리오 18: 승리를 즉시 확정하는 위험한 적재

**상황**

- 팀 점수가 96이고 수량 4 배터리를 외부 창고에 적재하면 즉시 목표 점수 100에 도달한다.

**예상 확률 변화**

- 게임 종료와 함께 \(z=+1\)이 발생한다.
- terminal potential은 0으로 처리한다.

**예상 보상**

- 팀 합 (+1)을 기준으로 명확한 큰 양수.
- 적재 직전 위험도는 더 이상 중요하지 않다. 실제 게임이 즉시 종료되기 때문이다.

**평가**

- 적절하다. 다음 흡수 확률만 사용하면 즉시 목표 점수 종료를 과소평가할 수 있으므로 terminal reward와 최종 승률 항이 반드시 필요하다.

### 시나리오 19: 크게 앞선 팀의 불필요한 위험 감수

**상황**

- 에피소드 종료가 가까우며 아군이 확정 점수로 크게 앞서 있다.
- Carrier가 적 창고 약탈을 시도하면서 사망 위험을 감수한다.

**예상 확률 변화**

- 얻을 수 있는 추가 점수의 승률 marginal은 작다.
- 실패로 직업 구성이나 방어가 무너지면 승률은 감소한다.

**예상 보상**

- 성공해도 작은 양수, 위험이 크면 기대상 음수.

**평가**

- 적절하다. 단순 점수 증가량이 아니라 승률의 포화 효과를 반영한다.

### 시나리오 20: 크게 뒤진 팀의 고위험 역전 시도

**상황**

- 안전한 플레이로는 시간 내 역전 가능성이 거의 없다.
- 성공 확률은 낮지만 적의 대형 외부 창고를 털면 역전 가능하다.

**예상 확률 변화**

- 보수적 행동의 승률은 거의 0에 머문다.
- 고위험 행동은 실패 확률이 높아도 승률 기댓값을 높일 수 있다.

**예상 보상**

- 승률이 실제로 증가한다면 양수.

**평가**

- 적절하다. 점수차만 사용하는 reward보다 상황에 맞는 risk-seeking 행동을 유도한다.

### 시나리오 21: 버프 아이템을 창고에 넣었다가 빼기

**상황**

- 속도 버프를 적재해 아군 운반 확률을 높인 뒤 다시 제거한다.

**예상 확률 변화**

- 적재 시 여러 배터리의 \(\pi_b^k\)와 승률이 증가한다.
- 제거 시 같은 변화가 반대로 돌아간다.

**예상 보상**

- 적재 때 양수, 제거 때 음수.
- 원래 상태로 돌아오는 반복 cycle의 누적 shaping은 거의 0.

**평가**

- 적절하다. 버프 아이템에 별도 고정 보상을 중복해서 주지 않아야 한다.

### 시나리오 22: 비대칭 맵에서 초기 위치 운이 좋음

**상황**

- episode 시작 시 한 팀 근처에 고가 배터리가 더 많이 생성된다.

**예상 확률 변화**

- 초기 \(\Psi(s_0)\)가 이미 유리한 값을 가질 수 있다.
- 초기 상태는 행동 결과가 아니므로 어떤 에이전트에게도 보상으로 지급하면 안 된다.

**예상 보상**

- 첫 행동 이전에는 0.
- 이후 변화량만 보상.

**평가**

- 적절하다. episode reset 직후 이전 episode의 potential과 차분하지 않도록 반드시 potential cache를 재초기화해야 한다.

---

## 10. 시나리오 평가에서 발견되는 남은 위험

### 10.1 확률 모델의 분포 외 상태

새 정책이 이전 데이터에서 거의 보지 못한 전략을 사용하면 확률 모델이 과신할 수 있다. 예를 들어 모든 유닛이 동시에 Hunter가 되는 상태가 학습 데이터에 없으면 승률을 잘못 예측할 수 있다.

대응 방법:

- ensemble 모델의 disagreement를 불확실성으로 사용
- 분포 외 상태에서는 shaping 계수를 축소
- \(\eta_{\mathrm{eff}}=\eta\exp(-c\,\mathrm{uncertainty})\) 적용
- 주기적으로 새 정책 rollout으로 모델 갱신

### 10.2 모델 exploitation

정책이 실제 승리가 아니라 예측 모델이 높게 평가하는 상태를 찾을 수 있다. 이를 방지하려면 terminal reward를 유지하고 shaping 크기를 \(\eta=0.25\) 정도로 제한해야 한다. 평가 때는 reward model 점수가 아니라 실제 승률을 사용한다.

### 10.3 reward non-stationarity

확률 모델을 매 학습 step 갱신하면 같은 transition의 reward가 변한다. 다음 반복 구조를 사용한다.

1. 현재 정책으로 self-play 데이터 수집
2. 귀속 확률 및 승률 모델 학습·교정
3. reward model 고정
4. 고정 모델로 정책을 일정 구간 학습
5. 모델 갱신 시 replay reward 재계산 또는 기존 buffer 폐기

초기 권장 고정 구간은 100,000 environment step이다.

### 10.4 동시 사건의 순서 의존성

Hunter 상호 사망, 사망과 아이템 파괴, 적재와 목표 점수 도달처럼 같은 physics step에 여러 사건이 생길 수 있다. 이벤트마다 reward를 계산하지 말고 모든 게임 로직이 끝난 뒤 하나의 원자적 \(s_t\to s_{t+1}\) 전이에 대해 reward를 한 번 계산해야 한다.

### 10.5 직업 선택권 누락

승률 모델이 현재 클래스만 보고 “Collector가 나중에 변환할 수 있다”는 사실을 모르면 조기 변환을 과대평가한다. 다음 feature를 명시적으로 넣는다.

- 에이전트별 도달 가능한 성소와 도달 시간
- 가능한 변환 집합
- 현재 Carrier 슬롯 lock 상태
- 죽음을 통한 Collector 복귀 예상 시간과 손실
- 팀 전체의 대체 가능한 Collector 수

---

## 11. 필수 검증 기준

### 11.1 zero-sum 검사

비종료 transition에서:

\[
|R_{A,t}+R_{B,t}|<10^{-5}
\]

가 되도록 같은 전역 확률분포에서 양 팀 효용을 계산한다.

### 11.2 순환 보상 검사

동일 전략 상태로 돌아오는 pickup/drop, storage steal/redeposit, buff insert/remove cycle의 discounted shaping 합이 수치 오차 범위에서 0에 가까워야 한다.

### 11.3 확률 calibration 검사

- 예측 0.7인 배터리의 실제 귀속률이 약 70%인지 확인
- 보호 창고 배터리는 거의 1로 예측되는지 확인
- 흡수까지 시간이 짧아질수록 다른 조건이 같을 때 생존 확률이 감소하지 않는지 확인
- 적 Hunter가 가까워질 때 운반자의 확정 확률이 증가하는 단조성 위반이 없는지 확인

### 11.4 직업 반사실 검사

고정 seed와 같은 상태 snapshot에서 클래스 하나만 바꾸어 Monte Carlo rollout을 반복한다.

\[
\widehat\Delta_i^{\mathrm{role}}
=\frac1M\sum_{m=1}^MZ_{k,m}^{c}
-\frac1M\sum_{m=1}^MZ_{k,m}^{\mathrm{Collector}}
\]

예측 모델의 \(\Delta_i^{\mathrm{role}}\)와 실제 Monte Carlo 차이의 부호가 일치하는지 확인한다. 특히 다음 상태군을 별도로 검사한다.

- Hunter 부족/과잉
- Carrier가 필요한 상태/불필요한 상태
- Carrier 슬롯을 잘못 선점하는 상태
- Collector가 한 명만 남은 상태
- 흡수 직전/직후
- 큰 점수 우세/열세

### 11.5 실제 정책 평가

최종 성공 기준은 reward return이 아니라 holdout seed에서의 실제 성과다.

- 실제 승률
- 최종 확정 점수차
- 배터리 수량 가중 pickup→absorb 성공률
- 약탈 후 회수 또는 상대 득점 차단 성공률
- 빈손 킬 비율
- 동일 아이템 반복 상호작용 횟수
- 직업별 평균 생존 시간과 직업 구성 분포
- Carrier 슬롯의 평균 기여도
- 외부 창고 배터리 생존 확률의 calibration

---

## 12. 초기 구현 파라미터 요약

```text
discount gamma                         = 0.99995
potential scale eta                    = 0.25
final win/draw/loss team reward        = +1 / 0 / -1
final per-agent base reward            = +1 / 0 / -1  (undivided broadcast to all 5 teammates, not divided by 5 — see §14.2, §16.1)

state potential:
  calibrated final outcome utility     = 0.60
  next-absorption superiority utility  = 0.40

direct pickup reward                   = 0
direct deposit reward                  = 0
direct absorption reward               = 0
fixed steal reward                     = 0
fixed kill reward                      = 0
fixed death penalty                    = 0
fixed Hunter transform reward          = 0
fixed Carrier transform reward         = 0

reward model freeze interval           = 100,000 environment steps initially
individual auxiliary TD coefficient   = 0.10 if individual credits are trained
```

---

## 13. 최종 판단

이 설계는 다양한 게임 사건을 하나의 일관된 단위로 환산한다.

\[
\boxed{
\text{사건의 가치}
=
\text{기대 점수 우세 변화}
+
\text{기대 승률 변화}
}
\]

- 운반과 적재는 배터리의 아군 확정 확률을 높인 만큼 보상된다.
- 약탈은 상대 확정 확률을 낮추고 아군 확정 확률을 높인 만큼 보상된다.
- 킬은 실제 배터리 흐름, 창고 방어 또는 직업 구성에 영향을 준 만큼만 보상된다.
- 사망은 고정 처벌이 아니라 잃은 아이템, 위치, 역할과 회복된 선택권의 순효과로 평가된다.
- 직업 변환은 현재 능력의 변화뿐 아니라 Collector의 미래 선택권과 Carrier 유일 슬롯의 기회비용까지 포함한 반사실 승률 차이로 평가된다.
- 크게 앞선 상태에서는 불필요한 위험의 marginal이 작아지고, 크게 뒤진 상태에서는 실제 역전 확률을 높이는 고위험 행동이 양의 가치를 가질 수 있다.

시나리오 분석상 목적 정렬성은 고정 이벤트 보상보다 높다. 가장 큰 구현 위험은 보상식 자체가 아니라 확률 모델의 calibration, 분포 외 과신, 직업 option value 누락 및 학습 중 reward non-stationarity이다. 따라서 보상 모델을 정책 학습 구간 동안 고정하고, counterfactual snapshot 평가와 holdout seed 실제 승률로 지속 검증하는 것이 필수다.

---

## 14. 검토 결과 및 Phase 1 구현 결정 사항

본 제안서 작성 후 실제 게임 코드(`blackout/Assets/Project/Runtime/Scripts`)와 대조 검증했고, 그 결과를 반영해 최초 구현(Phase 1) 범위를 아래와 같이 확정한다.

### 14.1 게임 메커니즘 대조 검증 결과

§1~§13에서 가정한 메커니즘은 다음과 같이 확인되었다.

| 가정 | 검증 결과 |
|---|---|
| 배터리 수량(`q_b`)이 점수에 비례 | 확인. `ItemObject.ItemAmount` = 점수 기여량 (`ScoreItemEffect.OnEnterStorage`) |
| 적재 후 주기적 흡수(`T_abs`) | 확인. `GameBalanceConfig.AbsorptionInterval`(기본 120s) 주기 이벤트. 단, 점수는 적재 즉시(잠정) 반영되고 흡수 시 확정됨 — 조건부 기댓값 프레이밍과 일치 |
| 보호 창고 vs 외부 창고 | **확인.** 본진 인접 고정 창고는 `MapTileData.TileCollisionOption.BlockEnemy` 타일로 둘러싸여 적 팀이 물리적으로 진입 불가(`MapManager.IsWalkable`). 절차 생성 외부 창고는 `Pass` 타일이라 양 팀 모두 접근 가능. `GameBalanceConfig.StorageCountPerTeam` 툴팁도 "protected base storage"라고 명시 |
| Collector→Hunter/Carrier 비가역, 사망시 복귀 | 확인 (`Sanctuary.TryTransform`, `MatchManager.RespawnUnit`) |
| Carrier 팀당 1슬롯 | 확인 (`Sanctuary.IsUniqueInstanceConstraint`) |
| 5v5, 0.02s tick, 사망시 아이템 파괴, 목표점수 100 즉시 종료 | 모두 확인 |
| 경로 pathfinding 인프라 | **미존재.** 코드베이스에 A*/BFS/NavMesh 없음. §3.2의 "최단 경로 시간" feature는 구현이 필요한 신규 인프라 |

### 14.2 Phase 1 범위 결정

풀 설계(학습된 배터리 귀속 분류기 `f_θ`, 승률 모델 `g_ω`, self-play 라벨링·calibration·freeze 루프, §7 반사실 개별 크레딧)는 구현 비용이 크고 cold-start 문제(1라운드에는 `f_θ`/`g_ω`가 없어 shaping이 불가능)가 있어, **Phase 1은 학습 모델 없이 상태의 결정론적 함수로만 potential을 정의**한다.

```
Ψ_k(s) = tanh( ( ConfirmedScoreDiff_k + Σ_b q_b · survive(b) · sign_b(k) ) / potentialScale )

survive(b) = 1                                   (b가 있는 창고 타일이 상대 팀에게 도달 불가능 — BlockEnemy 연결성으로 판정)
           = exp( -hazardCoefficient / max(d_enemy, 0.5) · τ )   (그 외: d_enemy = 최근접 적까지 직선거리, τ = 다음 흡수까지 남은 시간)
```

- `sign_b(k)`: 배터리를 현재 보유(운반 중) 또는 보관(창고) 중인 팀이 k면 +1, 상대 팀이면 -1. 필드에 방치된 미획득 배터리는 v1에서 기여 0 (향후 확장 가능).
- 승률 모델 `g_ω` 및 §6의 `0.6·U_win + 0.4·U_abs` 가중은 Phase 1에서 사용하지 않음. `Ψ = U_abs`에 해당하는 항만 사용.
- 거리 `d_enemy`는 §14.1에서 확인했듯 pathfinding이 없으므로 **직선거리로 근사**한다. 추후 그리드 BFS 기반 팀별 경로시간으로 교체 예정(§14.1 마지막 행).
- "보호 창고 도달 불가능" 판정은 각 팀 스폰에서 시작한 그리드 BFS 연결성(`IsWalkable` 기반)으로 계산한다. 이는 §3.2의 완전한 최단 경로 pathfinding과 다른, 도달 가능 여부만 판정하는 단순 서브셋이다.
- §7 반사실 개별 크레딧 배분은 보류. 기존 코드가 이미 팀 성과를 5명 전원에게 동일하게 분배(undivided broadcast)하는 방식을 쓰고 있으므로(`OnGameEnded`, 기존 kill/death/item 보상), 이 shaping 항도 같은 컨벤션을 따라 팀당 계산된 값을 5명 각각에게 그대로(=나누지 않고) 지급한다.
- 기존 고정 보상(`killReward`, `deathPenalty`, `itemRewards`, `teamScoreReward/Penalty`)은 §12 결정대로 0으로 설정. terminal ±1/0 및 γ, η 구조는 §6 그대로 유지.
- Non-stationarity/freeze 루프(§10.3), calibration 검증(§11.3)은 Phase 1에는 해당 없음 — potential이 학습되지 않는 순수 함수이므로 고정 문제 자체가 없음. 대신 hazard 계수 등은 경험적으로 튜닝해야 하는 하이퍼파라미터로 남는다.

### 14.3 Phase 2 (보류)

- §3.2 전체 feature를 쓰는 학습된 `f_θ`/`g_ω` 및 §10.3 freeze 루프
- §7 반사실 개별 크레딧(`d_i`) 및 individual auxiliary TD loss
- 직선거리 대신 팀별 최단 경로 시간을 쓰는 정식 pathfinding
- 필드에 있는 미획득 배터리의 귀속 확률 모델링

---

## 15. Phase 1.5: 개인별 내비게이션 potential (기초 fetch→carry 행동 보조 shaping)

### 15.1 배경 및 문제

§14.2의 \(\Psi_k(s)\)는 §14.3에서 명시했듯 **필드에 방치된 미획득 배터리를 potential에 전혀 반영하지 않는다** (`sign_b(k)`가 정의되지 않음 — 아무도 소유하지 않은 상태이므로). 그 결과 유닛이 미획득 아이템을 향해 이동하는 구간에는 어떤 dense shaping도 존재하지 않고, 오직 실제로 주운 순간(운반 상태로 전환되어 \(\Psi\)가 불연속적으로 변하는 순간)에만 보상이 발생한다. §12의 `direct pickup reward = 0` 결정과 결합하면, "아이템까지 이동" 구간은 현재 구현에서 완전히 무보상 구간이 되어 탐색만으로 발견해야 하는 sparse-reward 문제가 생긴다.

§9 시나리오 8은 이 공백을 고정 pickup 보상으로 메우는 것을 명시적으로 배제한다 ("별도의 고정 pickup 보상을 추가하면 이 성질[순환 보상 방지]이 깨진다"). 따라서 보완책도 §6과 동일한 **potential-based shaping(PBRS)** 형태여야 하며, 이벤트 트리거형 고정 보상이어서는 안 된다.

본 절은 §3.2의 학습된 \(f_\theta\) 없이, \(\Psi_k\)와 별개의 **에이전트 로컬(개인) potential** \(\Phi_i(s)\)를 정의해 이 공백을 메운다. \(\Psi_k\)가 "팀 관점의 확정/잠정 점수 우세"를 측정하는 반면, \(\Phi_i\)는 "유닛 \(i\)가 자신의 현재 목표(미획득 아이템 또는 귀속용 창고)에 얼마나 가까운가"만 측정하는, 순수하게 개인적이고 팀 zero-sum과 무관한 보조 채널이다.

### 15.2 개인 상태 potential 정의

유닛 \(i\)에 대해 다음 지시 변수를 둔다.

\[
C_i\in\{0,1\}:\ \text{Collectable 여부 (}\texttt{UnitData.Collectable}\text{)}, \qquad
h_i\in\{0,1\}:\ \text{아이템 보유 여부 (}\texttt{HoldingItem}\neq\text{null}\text{)}
\]

목표까지의 거리 함수를 정의한다. \(p_i\)는 유닛 \(i\)의 월드 좌표.

\[
d_i^{\mathrm{fetch}}(s)=
\begin{cases}
\displaystyle\min_{j\in U_i(s)}\lVert p_i-p_j\rVert_2 & U_i(s)\neq\emptyset\\[4pt]
D_\infty & U_i(s)=\emptyset
\end{cases}
\]

\[
U_i(s)=\{\text{아이템 } j : \texttt{State}_j=\texttt{OnGround},\ \texttt{OwnedTile}_j\text{가 어느 Storage에도 속하지 않음},\ \texttt{IsInteractable}(j,\ \mathrm{Team}_i)\}
\]

\[
d_i^{\mathrm{carry}}(s)=
\begin{cases}
\displaystyle\min_{\tau\in S_i(s)}\lVert p_i-c(\tau)\rVert_2 & S_i(s)\neq\emptyset\\[4pt]
D_\infty & S_i(s)=\emptyset
\end{cases}
\]

\[
S_i(s)=\{\text{타일 } \tau\in\text{(유닛 }i\text{ 소속 팀의 Storage)} : \tau\text{가 현재 유닛 }i\text{가 든 아이템을 받을 수 있음(빈 타일이거나 같은 종류이고 수량 미포화)}\}
\]

\(c(\tau)\)는 타일 \(\tau\)의 월드 중심 좌표(`MapManager.CellToCenterWorld`), \(D_\infty\)는 충분히 큰 sentinel 거리(구현에서는 `navPotentialScale`의 8배 — 아래 포화 함수가 그 지점에서 이미 0에 수렴하므로 "타깃 없음"에 대한 별도 분기 없이 자연스럽게 처리된다).

거리를 \([0,1]\)로 포화시키는 함수:

\[
\varphi(d)=1-\tanh\!\left(\frac{d}{L_{\mathrm{nav}}}\right),\qquad L_{\mathrm{nav}}=\texttt{navPotentialScale}
\]

개인 potential:

\[
\boxed{
\Phi_i(s)=C_i\cdot\Big[(1-h_i)\,\varphi\big(d_i^{\mathrm{fetch}}(s)\big)+h_i\,\varphi\big(d_i^{\mathrm{carry}}(s)\big)\Big]
}
\]

\(C_i=0\)(Hunter류)이면 \(\Phi_i\equiv 0\)으로 고정되어 전투/수비 역할의 행동을 왜곡하지 않는다. \(\Phi_i\in[0,1]\)이며, 목표에 도달할수록(거리→0) 1에 가까워지고 목표가 없거나 멀수록 0에 가까워진다.

거리는 §14.2의 \(d_{\mathrm{enemy}}\)와 동일하게 **직선거리로 근사**한다 (§14.1에서 확인된 pathfinding 인프라 부재와 동일 제약).

### 15.3 보상식

\[
\boxed{
r_{i,t}^{\mathrm{nav}}=\eta_{\mathrm{nav}}\big[\gamma\,\Phi_i(s_{t+1})-\Phi_i(s_t)\big]
}
\]

할인율 \(\gamma\)는 §14.2의 `potentialGamma`를 그대로 재사용한다(별도 파라미터를 늘리지 않기 위함). \(\eta_{\mathrm{nav}}=\texttt{navPotentialEta}\)는 §6의 \(\eta\)와 분리된 별도 스케일이며, 이 항이 "메인 목표가 아닌 탐색 보조 신호"임을 반영해 \(\eta_{\mathrm{nav}}\ll\eta\)로 둔다(§15.6 참고).

이 보상은 \(\Psi_k\) 기반 shaping(§14.2, 팀 5명에게 undivided broadcast)과 달리 **그 유닛 자신에게만** 지급한다 — 기존 kill/pickup 이벤트 보상(`BlackOutAgent.cs`)과 같은 개인 타게팅 방식이다.

### 15.4 팀 zero-sum과의 관계

\(\Psi_k\)는 \(\Psi_{-k}=-\Psi_k\)로 정의되어 §11.1의 팀간 zero-sum 검사(\(|R_{A,t}+R_{B,t}|<10^{-5}\))를 만족해야 한다. \(\Phi_i\)는 **이 검사의 대상이 아니다** — 각 팀은 자신의 유닛 상태만으로 독립적으로 \(\Phi_i\)를 계산하므로, 한 팀 근처에 아이템이 더 많이 생성되면(§9 시나리오 22와 유사한 비대칭) 그 팀의 \(\sum_i r_i^{\mathrm{nav}}\)이 더 커질 수 있다. 이는 §11.1이 막으려는 "팀 성과와 무관한 보상 누수"가 아니라, 실제 그 팀이 처한 상황을 반영한 개인 내비게이션 신호이므로 zero-sum 제약 밖에 둔다. §11.1 검증 스크립트는 \(\Psi_k\) 기반 shaping 항만 대상으로 하고, \(\Phi_i\) 항은 별도로 §15.5의 순환 보상 검사만 적용한다.

### 15.5 안전성 (순환 보상 및 정책 불변성)

\(\Phi_i:S\to[0,1]\)는 행동이 아닌 상태만의 함수이므로, 임의의 궤적 \(s_0,\dots,s_T\)에 대해 다음 telescoping 항등식이 성립한다.

\[
\sum_{t=0}^{T-1}\gamma^t\eta_{\mathrm{nav}}\big[\gamma\Phi_i(s_{t+1})-\Phi_i(s_t)\big]
=\eta_{\mathrm{nav}}\big[\gamma^T\Phi_i(s_T)-\Phi_i(s_0)\big]
\]

\(s_T=s_0\)인 순환 궤적(아이템을 반복해서 접근·이탈하거나 줍고 놓는 경우, §9 시나리오 8·21과 동일 패턴)에서는 다음과 같다.

\[
\eta_{\mathrm{nav}}\big[\gamma^T\Phi_i(s_0)-\Phi_i(s_0)\big]=\eta_{\mathrm{nav}}\Phi_i(s_0)(\gamma^T-1)
\]

이 값은 반복 횟수 \(T\)에 비례해 커지지 않고 \(\gamma^T\to0\)에 따라 \(-\eta_{\mathrm{nav}}\Phi_i(s_0)\)로 수렴하는 **유계(bounded)** 값이다. 즉 같은 사이클을 몇 번을 반복하든 누적 shaping 총량은 \(\eta_{\mathrm{nav}}\)와 \(\Phi_i(s_0)\le1\)로 유계이며, 반복으로 무한히 양의 보상을 축적할 수 없다(§11.2 순환 보상 검사와 동일 기준으로 검증 가능).

또한 Ng, Harada & Russell(1999)의 PBRS 불변성 정리에 따라, terminal potential을 0으로 두면(기존 `ApplyPotentialShaping`이 `CurrentState != Playing`일 때 shaping을 멈추는 것과 동일한 근사) 이 항은 최종 수렴 정책 자체를 바꾸지 않고 학습 속도만 개선한다는 이론적 보장이 있다 — 단, 이는 \(\Phi_i\)가 학습 중 고정된 함수일 때만 성립하며, 본 설계는 §14.2와 마찬가지로 학습되지 않는 순수 함수이므로 이 조건을 만족한다.

### 15.6 파라미터

```text
navPotentialEta      = 0.08   (potentialEta=0.25보다 작게 — 보조 신호이지 메인 목표가 아님)
navPotentialScale    = 12.0   (타일 단위 거리 정규화, 맵 크기의 절반 정도)
navPotentialGamma    = potentialGamma 재사용 (별도 파라미터 미도입)
```

기초 fetch→carry 행동이 안정적으로 학습된 이후에는 \(\eta_{\mathrm{nav}}\)를 점진적으로 0으로 낮추는 curriculum을 권장한다 — §10.3의 "고정 구간 후 교체" 컨벤션과 같은 정신으로, 이 항은 학습 초기의 보조 바퀴이지 최종 목적 함수의 일부가 아니다.

### 15.7 범위 제한 및 후속 과제

- 적 창고에서의 절도(steal) 타깃은 \(U_i(s)\)에 포함하지 않는다. hazard/경로 판단 없이 거리만으로 접근을 유도하면 무모한 돌진을 부추길 수 있어, §3.2의 위험도 feature가 갖춰지기 전까지는 제외한다.
- 직선거리는 §14.1·§14.3과 동일하게 향후 그리드 BFS 기반 경로시간으로 교체 대상이다.
- \(\Phi_i\)는 §7의 반사실 개별 크레딧(\(d_i\))을 대체하지 않는다 — "누가 상태 개선을 일으켰는지"의 엄밀한 인과 추정이 아니라 "지금 내가 목표에 가까운가"라는 근사적 개인 신호이며, QMIX 트레이너가 팀 보상을 합산(`qmix_trainer.py`의 `reward_a = sum(rewards[team_a])`)하므로 팀 단위로 보면 \(\sum_i\Phi_i(s)\) 역시 유효한 하나의 potential 함수로서 §15.5의 안전성 논증이 합산 후에도 그대로 유지된다.

---

## 16. 구현 검증 현황 (2026-09-12 기준)

실제 코드(`blackout/Assets/Project/Runtime/Scripts/ML/`)와 대조 검증한 결과를 기록한다. Phase 1(§14.2)과 Phase 1.5(§15)에 명시된 항목은 아래에서 확인된 대로 **모두 구현되어 있다.** Phase 2(§14.3) 및 §7 반사실 크레딧은 설계대로 아직 보류 상태다.

### 16.1 Phase 1 (§14.2) — 구현 확인됨

- \(\Psi_k(s)=\tanh\!\big((\mathrm{ConfirmedScoreDiff}+\sum_bq_b\cdot\mathrm{survive}(b)\cdot\mathrm{sign}_b(k))/\mathrm{potentialScale}\big)\): [PotentialRewardCalculator.cs:37-55](../../../blackout/Assets/Project/Runtime/Scripts/ML/PotentialRewardCalculator.cs)에서 수식 그대로 구현.
- `survive(b)`: 보호 창고(적 팀 스폰에서 BFS로 도달 불가)이면 1, 그 외에는 최근접 적까지 **그리드 경로 거리** 기반 `exp(-hazard/d · τ)`. **2026-09-12 갱신**: 기존 직선거리 근사를 [GridPathfinder.cs](../../../blackout/Assets/Project/Runtime/Scripts/ML/GridPathfinder.cs)의 8방향 코너컷 방지 최단경로 탐색으로 교체 — 벽 너머의 적을 더 이상 "가깝다"고 과대평가하지 않는다. [PotentialRewardCalculator.cs:97-156](../../../blackout/Assets/Project/Runtime/Scripts/ML/PotentialRewardCalculator.cs)
- 미획득 필드 배터리는 potential에 기여 0 (`EnumerateActiveBatteries`가 보유/보관 중인 아이템만 순회). [PotentialRewardCalculator.cs:62-86](../../../blackout/Assets/Project/Runtime/Scripts/ML/PotentialRewardCalculator.cs)
- \(R_{k,t}=z_{k,t}+\eta[\gamma\Psi_k(s_{t+1})-\Psi_k(s_t)]\), \(\gamma=0.99995\), \(\eta=0.25\), terminal potential 0 (게임 종료 시 `ApplyPotentialShaping`이 조기 반환): [BlackOutEpisodeCoordinator.cs:142-158](../../../blackout/Assets/Project/Runtime/Scripts/ML/BlackOutEpisodeCoordinator.cs), 파라미터는 [reward_config.json](../../../blackout/Assets/StreamingAssets/reward_config.json)에서 확인.
- \(\Psi_{-k}=-\Psi_k\) zero-sum: `prevPsiA` 하나만 계산해 양 팀에 부호만 바꿔 적용하므로 구조적으로 zero-sum이 보장됨. [BlackOutEpisodeCoordinator.cs:146-157](../../../blackout/Assets/Project/Runtime/Scripts/ML/BlackOutEpisodeCoordinator.cs)
- 고정 보상(`killReward`/`deathPenalty`/`itemRewards`/`teamScoreReward`/`teamScorePenalty`)은 런타임 설정 파일에서 모두 0: [reward_config.json](../../../blackout/Assets/StreamingAssets/reward_config.json). 이벤트 발생 경로(`BlackOutAgent.cs`) 자체는 그대로 남아있고 값만 0으로 배선됨 — 향후 다시 값을 넣기만 하면 재활성화 가능한 구조.
- Ψ 기반 shaping은 §14.2가 정한 대로 5명에게 **나누지 않고** 그대로(undivided broadcast) 지급됨: [BlackOutEpisodeCoordinator.cs:151-157](../../../blackout/Assets/Project/Runtime/Scripts/ML/BlackOutEpisodeCoordinator.cs).
- **§12 표 수정**: terminal 팀 보상 ±1도 같은 undivided 컨벤션으로 5명 전원에게 그대로 지급됨 (`OnGameEnded`가 `reward = winner==team ? 1f : -1f`를 각 agent에 그대로 적용). [BlackOutEpisodeCoordinator.cs:206-218](../../../blackout/Assets/Project/Runtime/Scripts/ML/BlackOutEpisodeCoordinator.cs) 기존 §12의 "±0.2(=±1÷5)" 표기는 실제 구현과 달라 위 표를 정정했다.

### 16.2 Phase 1.5 (§15) — 구현 확인됨

- \(\Phi_i(s)=C_i\cdot[(1-h_i)\varphi(d_i^{\mathrm{fetch}})+h_i\varphi(d_i^{\mathrm{carry}})]\), \(\varphi(d)=1-\tanh(d/L_{\mathrm{nav}})\): [IndividualNavPotentialCalculator.cs:28-40](../../../blackout/Assets/Project/Runtime/Scripts/ML/IndividualNavPotentialCalculator.cs)
- \(U_i(s)\)(미획득·비창고·상호작용 가능 아이템)와 \(S_i(s)\)(자팀 창고 중 수용 가능 타일) 판정, \(D_\infty=8\times\)`navPotentialScale` sentinel: [IndividualNavPotentialCalculator.cs:19-93](../../../blackout/Assets/Project/Runtime/Scripts/ML/IndividualNavPotentialCalculator.cs). **2026-09-12 갱신**: \(d_i^{\mathrm{fetch}}\)/\(d_i^{\mathrm{carry}}\)도 직선거리 대신 같은 `GridPathfinder` 기반 그리드 경로 거리로 계산하며, 목표가 있어도 도달 불가능하면(막힌 구역) "타깃 없음"과 동일하게 처리된다.
- §15.7의 적 창고 절도 타깃 제외: 창고에 속한 타일의 아이템은 팀 무관하게 fetch 타깃에서 제외됨(같은 필터로 아군/적군 창고 모두 배제) — 동일 파일.
- \(r_{i,t}^{\mathrm{nav}}=\eta_{\mathrm{nav}}[\gamma\Phi_i(s_{t+1})-\Phi_i(s_t)]\), \(\eta_{\mathrm{nav}}=0.08\), \(L_{\mathrm{nav}}=12.0\), \(\gamma\)는 `potentialGamma` 재사용, 해당 유닛에게만 개별 지급(팀 broadcast 아님): [BlackOutEpisodeCoordinator.cs:166-181](../../../blackout/Assets/Project/Runtime/Scripts/ML/BlackOutEpisodeCoordinator.cs), [reward_config.json](../../../blackout/Assets/StreamingAssets/reward_config.json)
- \(C_i=0\)(Hunter류)이면 \(\Phi_i\equiv0\): `Collectable` 게이트로 구현. [IndividualNavPotentialCalculator.cs:30](../../../blackout/Assets/Project/Runtime/Scripts/ML/IndividualNavPotentialCalculator.cs)

### 16.3 확인된 미구현/괴리 항목

- **§7 반사실 개별 크레딧(\(d_i\)) 및 individual auxiliary TD loss**: Phase 2로 명시 보류, 미구현. `qmix_trainer.py`는 여전히 팀원 5명의 보상을 합산한 스칼라 하나만 replay buffer에 저장한다.
- ~~**직선거리 근사**~~ — **2026-09-12 부분 해결.** `survive(b)`의 hazard 거리와 \(\Phi_i\)의 \(d^{\mathrm{fetch}}/d^{\mathrm{carry}}\) 모두 [GridPathfinder.cs](../../../blackout/Assets/Project/Runtime/Scripts/ML/GridPathfinder.cs)의 그리드 최단경로 탐색으로 교체했다(§16.1/§16.2 참고). "가장 가까운 타깃"을 찾는 질의이므로 후보마다 개별 A*를 도는 대신, 출발점에서 한 번 다익스트라/A*(휴리스틱 0)를 흘려보내 처음 만나는 타깃에서 멈추는 방식으로 구현 — 결과적으로 8방향·코너컷 방지 규칙까지 포함해 후보별 A*와 동일한 최단거리를 훨씬 적은 탐색으로 얻는다. **남은 근사**: 그리드 타일 단위 이동만 반영하고(연속 이동 자체는 모델링 안 함), §14.3이 원래 요구한 "팀별 최단 경로 **시간**"(속도/충돌크기/버프 반영)까지는 아니며 여전히 순수 거리 기준이다.
- **§14.3 Phase 2 항목 나머지** (학습된 \(f_\theta\)/\(g_\omega\), freeze 루프, 미획득 배터리 귀속 모델링, 경로 "시간" 변환): 코드베이스에 해당 인프라 없음 — 설계 문서 그대로 보류 상태 유지.
- **§11.1(zero-sum 검사) 자동 검증 테스트**: 여전히 미구현. Ψ의 zero-sum은 `prevPsiA` 단일 계산 구조상 보장되지만 수치로 확인하는 테스트는 아직 없다.
- ~~**§11.2/§15.5(순환 보상 검사) 자동 검증 테스트**~~ — **2026-09-12 추가 완료.** [tests/test_potential_shaping_math.py](../../tests/test_potential_shaping_math.py)에 순수 수식 기반 pytest 스타일 테스트 4개를 추가하고 실행까지 확인했다(Unity 불필요, `./.venv/bin/python tests/test_potential_shaping_math.py`로 직접 실행 가능). 임의의 potential 궤적에 대해 discounted return이 \(\eta[\gamma^T\Psi_T-\Psi_0]\)로 정확히 telescoping됨을 확인하고, 동일 길이 \(T\) 안에서 같은 cycle을 더 자주 반복해도(반복 횟수 \(N\)을 4배로) discounted return이 거의 변하지 않음을 확인했으며, γ<1일 때만 발생하는 raw(비할인) reward 합의 미세한 누수가 반복 횟수가 아니라 **에피소드 길이 \(T\)에만 비례해 유계**임(현재 파라미터로 최대 약 0.05 미만)을 수치로 보였다. 이는 게임 자체가 에피소드 길이를 420초로 제한하므로 실질적인 익스플로잇 경로가 없음을 뜻한다. 단, 이 테스트는 shaping 공식 자체의 수학적 성질만 검증하며 실제 `BlackOutEpisodeCoordinator.cs`가 그 공식을 정확히 그대로 호출하는지는 별도로 실증 검증(§16.4)해야 한다 — 마침 그 파일은 지금 다른 세션이 배속 문제로 수정 중이라 직접 대조는 보류했다.
- ~~**`RewardConfig.cs` 기본값 위험**~~ — **2026-09-12 수정 완료.** 클래스 필드 기본값(`killReward`/`deathPenalty`/`teamScoreReward`/`teamScorePenalty`)을 0으로 맞추고, `reward_config.json` 로드 실패 시 로그를 `LogWarning`에서 `LogError`로 올려 더 눈에 띄게 했다. 이제 config 파일이 어떤 이유로든 로드에 실패해도 폴백값 자체가 §12 스펙(고정 보상 0)과 일치하므로 고정 보상이 조용히 재도입될 수 없다. [RewardConfig.cs:19-29,60-71](../../../blackout/Assets/Project/Runtime/Scripts/ML/RewardConfig.cs)
- ~~**미등록 아이템의 `GetItemReward` 폴백**~~ — **2026-09-12 수정 완료.** `GetItemReward(name, fallback=0.1f)`의 기본 fallback을 `0f`로 변경했다. 이제 새 아이템 타입을 추가하면서 `itemRewards`에 등록을 누락해도 §12를 벗어난 고정 보상이 조용히 생기지 않는다(명시적으로 0이 아닌 값을 주고 싶다면 `itemRewards`에 등록하거나 호출부에서 `fallback` 인자를 명시해야 한다). [RewardConfig.cs:46](../../../blackout/Assets/Project/Runtime/Scripts/ML/RewardConfig.cs)

### 16.4 실증 평가 결과 (2026-09-12, 환경 배속 수정 반영 후 실행 완료)

다른 세션의 환경 배속 수정(커밋 `859e4b1`/`6b6ae85`/`d9140fe`/`4685112`)이 병합된 뒤 `CIBuild.BuildBlackOutMac`으로 재빌드하고, 기존 [evaluate_reward_model.py](../../examples/evaluate_reward_model.py)에 더해 미시/중간/거시 세 단위를 동시에 보는 [evaluate_reward_alignment.py](../../examples/evaluate_reward_alignment.py)를 새로 작성해 실행했다. `StrategicHeuristicV1`/`V4`/`V6`/`V8` 조합으로 headless, `time_scale=200`, 여러 Unity 인스턴스 병렬 실행(`--workers`)으로 진행했으며, 코드 수정 없이 `obs["agent_states"]`(위치·소지 배터리량)와 `infos`(score_0/score_1 등)만으로 이벤트를 감지했다.

**거시(승패) — 양호.** 결정된(무승부 아닌) 20~30경기 기준, 종료 전 누적 shaped return의 부호가 실제 승자와 77~80% 일치하고 score 차이와의 상관계수는 0.69~0.72. 터미널 ±1은 설계상 100% 일치(항등적 sanity check).

**미시(배터리 pickup/deposit) — Ψ 단독으로는 매우 양호, Φ_i가 이를 훼손.** 개인 내비게이션 potential을 끈 psi_only 조건에서 pickup 시 96.6~96.8%, deposit 시 84.4~85.3%가 기대 부호(+)와 일치했다. 그런데 Φ_i까지 포함한 실제 운영 설정(full)에서는 이 값이 pickup 82.9~83.7%, deposit 73.5~74.4%로 **뚜렷이 나빠진다** — 원인은 §16.5-1 참고.

**중간(score 변화 시점) — 표면적으로는 나쁘지만 원인은 관측 타이밍 아티팩트.** score_0/score_1이 바뀌는 틱(배터리 저장/탈취, n≈1550건)에서 같은 틱의 팀 shaped reward 부호 일치율이 42.6~45.3%(무작위 이하), 상관계수 -0.27로 나왔다. reward를 한 틱 뒤로(offset=+1) 밀어서 재계산하면 상관계수가 +0.56~+0.57로 반전되고 부호 일치율도 49.4~52.5%로 개선된다 — 원인과 해석은 §16.5-2 참고. 즉 shaping 공식 자체가 방향을 반대로 주는 것이 아니라, **관측치 두 스트림(파이썬이 보는 score와 reward)이 구조적으로 한 틱 어긋나 있어서** 순진하게 같은 틱끼리 비교하면 오히려 반대로 보인다. 다만 오프셋을 맞춰도 부호 일치율이 50% 안팎에 머무는 것은, 10명이 동시에 행동하는 상황에서 한 틱에 여러 배터리 이벤트가 겹쳐 노이즈가 크기 때문으로 보인다(추가 검증 필요, 아래 미해결 항목 참고).

### 16.5 새로 발견된 항목 (2026-09-12, 실증 평가로 확인)

1. ~~**Φ_i의 fetch/carry 목표 전환이 불연속이라 서브골 달성 순간에 오히려 음의 보상을 줄 수 있음**~~ — **2026-09-12 pickup 쪽 수정 완료, deposit 쪽은 의도된 동작으로 재확인.** 원래 [IndividualNavPotentialCalculator.cs](../../../blackout/Assets/Project/Runtime/Scripts/ML/IndividualNavPotentialCalculator.cs)의 `ComputePotential`은 `HoldingItem == null` 여부로 "가장 가까운 미획득 아이템까지 거리"와 "가장 가까운 자팀 창고까지 거리" 사이를 즉시 전환했다. §16.4에서 측정한 pickup/deposit 부호 일치율이 psi_only(96~97%/84~85%) 대비 full(83~84%/73~74%)에서 나빠지는 원인이 바로 이 전환이었다.
   - **pickup 쪽은 실제 결함이었다**: 빈손일 때의 potential이 "아이템까지 거리"만 봐서, 아이템에 도달하는 순간 아직 창고까지 옮기는 일이 남았는데도 potential이 최댓값 근처로 착각하게 만들고, 실제로 주운 직후 창고까지의 진짜 거리로 재계산되면서 potential이 급락했다. 수정: 빈손 potential을 **"아이템까지 거리 + 그 아이템 위치에서 창고까지의 거리"의 합**으로 재정의했다(\(d_i^{\mathrm{fetch}}\)를 §15.2의 정의에서 배달 전체의 잔여 거리로 확장). 줍는 순간 유닛 위치와 아이템 위치가 같으므로 이 합과 direct-후 carry-potential이 정확히 일치해 불연속이 사라진다. [IndividualNavPotentialCalculator.cs:67-113](../../../blackout/Assets/Project/Runtime/Scripts/ML/IndividualNavPotentialCalculator.cs), 거리 조회는 [GridPathfinder.cs](../../../blackout/Assets/Project/Runtime/Scripts/ML/GridPathfinder.cs)에 추가한 `NearestMatching`(어떤 타깃 셀이 매칭됐는지도 반환하는 버전)을 사용.
   - **deposit 쪽은 의도된 동작으로 재확인했다**: 창고에 막 내려놓은 직후 potential이 "다음 아이템까지 거리"로 재설정되어 하락하는 것은, 실제로 새로운(그리고 대개 더 먼) 배달이 막 시작됐다는 사실을 정확히 반영하는 것이지 모델링 오류가 아니다 — PBRS의 최적 정책 불변성(Ng et al. 1999)은 potential의 불연속 자체를 문제 삼지 않으므로, 이 하락은 그대로 두었다. 즉 §16.4에서 관찰된 deposit 부호 불일치(73~85%)의 상당 부분은 "리워드가 틀렸다"가 아니라 "새 배달이 시작돼서 남은 거리가 실제로 늘었다"는 정상적인 신호일 가능성이 높다 — 재평가 시 이 점을 감안해 해석해야 한다.
2. **MapObsAgent와 BlackOutAgent의 결정 주기 불일치로 인한 관측/보상 전달 타이밍 어긋남 — 리워드 공식 결함은 아니지만 문서화 필요.** [MapObsAgent.cs:67](../../../blackout/Assets/Project/Runtime/Scripts/ML/MapObsAgent.cs)는 `FixedUpdate()`마다 `RequestDecision()`을 호출해 매 틱 score를 갱신하지만, [BlackOutAgent.cs:28](../../../blackout/Assets/Project/Runtime/Scripts/ML/BlackOutAgent.cs)의 `decisionPeriod=2`는 유닛 에이전트가 2틱에 한 번만 그동안 누적된 보상을 파이썬에 전달하게 한다. 그 결과 파이썬 쪽에서 "이 스텝의 score"와 "이 스텝의 reward"를 같은 틱 인덱스로 나란히 비교하면 최대 1틱까지 어긋난다(§16.4의 offset 실험 참고). 학습 자체에는 영향이 없다(에이전트가 받는 누적 보상 총량과 그 원인이 된 행동 사이의 대응은 ML-Agents가 정확히 관리) — 다만 사람이 디버깅하거나 스텝 단위 보조 손실을 설계할 때 이 지연을 모르면 리워드 모델이 반대로 작동한다고 오판할 수 있다.

**미해결 후속 과제**: §11.1 zero-sum 자동 검증 테스트는 여전히 없음(§16.3에서 이미 지적). Φ_i 불연속과 중간 단위 잔여 노이즈는 아래 §16.6에서 후속 검증을 마쳤다.

### 16.6 후속 검증 (2026-09-13): Φ_i 불연속 수정 및 중간 단위 노이즈 원인 규명

**Φ_i 불연속 수정.** §16.5-1에서 지적한 pickup 쪽 불연속을 실제로 고쳤다. `NearestUnclaimedItemDistance`(빈손일 때 potential)를 "가장 가까운 아이템까지 거리"에서 **"가장 가까운 아이템까지 거리 + 그 아이템의 위치에서 가장 가까운 수용 가능 창고까지 거리"**로 재정의했다 — 줍는 순간 유닛 위치와 아이템 위치가 같아지므로 이 합이 pickup 직후의 carry-phase potential과 정확히 일치해 불연속이 사라진다. [GridPathfinder.cs](../../../blackout/Assets/Project/Runtime/Scripts/ML/GridPathfinder.cs)에 매칭된 타깃 셀도 함께 반환하는 `NearestMatching`을 추가했고, [IndividualNavPotentialCalculator.cs:67-154](../../../blackout/Assets/Project/Runtime/Scripts/ML/IndividualNavPotentialCalculator.cs)에서 이를 사용해 아이템 자신의 창고까지 거리를 조회한다(공용 헬퍼 `NearestAcceptingStorageDistance`로 carry-phase와 통합). 효과 검증(동일 시드로 재실행, 휴리스틱은 리워드를 보지 않으므로 궤적은 수정 전후 완전히 동일): pickup 부호 일치율이 full 설정 기준 82.9%→**96.0%**로 개선되어 psi_only(96.8%)와 거의 같아졌다. deposit 쪽은 오히려 73.5%→67.8%로 낮아졌는데, 이는 결함이 아니라 fetch potential이 이제 남은 일을 더 정직하게(더 낮게) 반영하면서 배달 직후의 정당한 하락폭이 커졌기 때문으로 해석한다 — §16.5-1에서 이미 "deposit의 하락은 새 배달이 시작된 것을 반영하는 정상 신호"라고 결론 내린 바와 일치한다.

**중간 단위 잔여 노이즈 원인 규명.** §16.4/§16.5에서 남겨둔 "offset 보정 후에도 부호 일치율이 50% 안팎에 머무는 이유"를 3가지 가설로 나눠 검증했다(10시드×3매치업=60경기, 8개 Unity 인스턴스 병렬 실행으로 스케일 확대):
1. **속성(팀 매핑) 오류** — 기각. score/reward 모두 "팀 A" 기준으로 일관되게 짝지어져 있음을 재확인했다.
2. **`MapObsAgent`(매 틱 결정) vs `BlackOutAgent`(`decisionPeriod=2`)의 관측/보상 전달 타이밍 어긋남** — §16.5에서 이미 확인한 대로 가장 큰 원인이며, offset=+1 보정으로 상관계수가 -0.26→+0.55 수준으로 반전됨을 60경기 규모에서도 재확인했다.
3. **동시다발 배터리 이벤트로 인한 노이즈** — 미미함. 전후 ±3틱 내 다른 score 변화가 없는 "고립 이벤트"만 걸러도 (n=3605/4536) 부호 일치율 49.1%, 상관계수 0.56으로 전체 풀링과 거의 동일했다. `psi_only`(개인 내비 potential 제거)로도 재검증했으나 53.8%/corr 0.58로 소폭 개선에 그쳐, "다른 유닛들의 내비게이션 노이즈"도 주된 원인은 아니었다.

잔여 노이즈의 가장 설득력 있는 설명은 Ψ가 score 외에도 `survive(b)`(가장 가까운 적까지의 그리드 거리 기반 hazard)에 의존하며, 이 값이 유닛들의 이동에 따라 **매 틱 연속적으로 변한다**는 점이다 — 이 "배경 변동"이 작은 배터리 하나의 한계 기여분보다 클 수 있어 개별 틱 단위 부호 비교를 흐린다. 다만 이는 부호가 아니라 **크기(magnitude)** 기준으로 보면 명확히 드러난다: |score 변화량|을 4분위로 나눠 평균 |reward|를 보면 0.031 → 0.062 → 0.127 → 0.160으로 단조 증가하고(전체 상관계수 0.30), 미시 단위에서도 pickup/deposit 시 배터리 보유량과 |reward|의 상관계수가 각각 0.63/0.42로 뚜렷한 양의 관계를 보인다. 참고로 `score_0`/`score_1`은 `TeamContext.Score`를 `TargetScore`(=100)로 정규화한 값이라([MapObsAgent.cs:104-106](../../../blackout/Assets/Project/Runtime/Scripts/ML/MapObsAgent.cs:104-106)), 실측 기울기(정규화 delta 0.01당 |reward| 0.0163, 즉 배터리 1개당 약 0.0163)가 Ψ의 score 항만으로 예측한 값(η/potentialScale=0.25/40=0.00625)의 약 2.6배인 것은 score와 provisionalDiff(점유 확률 항)가 같은 이벤트에서 동시에 같은 방향으로 움직이기 때문으로 정성적으로 설명된다.

**결론**: 중간 단위는 부호(방향)와 크기(스케일) 모두 근본적 결함이 아니라 "여러 정당한 요인이 겹치는 연속적 potential 함수의 자연스러운 특성"으로 판단하며, 추가 코드 수정 없이 이 상태로 충분하다고 본다. 검증에 사용한 스크립트는 [blackout-env/examples/evaluate_reward_alignment.py](../../examples/evaluate_reward_alignment.py) (`analyze_meso_offsets`/`analyze_meso_clean`/`analyze_meso_magnitude`/`analyze_micro`의 `corr_amount_vs_abs_reward`).
