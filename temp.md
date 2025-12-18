# 2025-01 컴퓨터소프트웨어캡스톤PBL 기말고사 문제 정리 (17~24)

> 🔴 **빨간 주석**: 출제 가능성 높은 핵심 포인트
> 
> ⭐ **별표**: 빈칸 출제 예상 지점

---

## 17. 클러스터 업그레이드 (이분 탐색)

### 문제 설명
대규모 머신 러닝에서는 여러 컴퓨터를 하나의 클러스터로 묶어 계산을 수행한다. 클러스터의 성능은 컴퓨터의 수가 많아질수록, 각각의 성능이 올라갈수록 향상된다.

클러스터에는 N대의 컴퓨터가 있으며, 각각의 성능은 $a_i$라는 정수로 평가할 수 있다. 성능을 $d$만큼 향상시키는 데에 드는 비용은 $d^2$원이다. (단, d는 자연수)

- 업그레이드를 하지 않는 컴퓨터가 있어도 됨
- 한 컴퓨터에 두 번 이상 업그레이드를 수행할 수 없음
- 예산 B원 이하의 총 비용으로 **성능이 가장 낮은 컴퓨터의 성능을 최대화**하는 것이 목표

### 제약 조건
- $1 \leq N \leq 10^5$ (정수)
- $1 \leq a_i \leq 10^9$ (정수)
- $1 \leq B \leq 10^{18}$ (정수, 64비트 정수형 필요)

### 입력 형식
```
N B
a₁ a₂ ... aₙ
```

### 출력 형식
```
예산을 효율적으로 사용했을 때, 성능이 가장 낮은 컴퓨터의 성능으로 가능한 최댓값
```

### 예제
**입력**
```
4 10
5 5 6 1
```
**출력**
```
4
```

### 풀이 전략
- **이분 탐색 (Binary Search)**: x가 커질수록 비용이 증가 → 단조성(monotonicity) 존재
- "비용 ≤ B"를 만족하는 최대 x를 이분 탐색으로 찾음

### 시간복잡도
$$O(N \log(\text{범위})) = O(N \log(2 \times 10^9))$$

### 정답 코드
```python
n, b = map(int, input().split())
a = list(map(int, input().split()))

def test(x):
    cost = 0
    for i in range(n):
        if a[i] < x:
            cost += (x - a[i]) * (x - a[i])  # ⭐(a) 비용 계산: 제곱!
    return cost <= b  # 🔴 판별 함수의 반환값

low, high = 1, 2 * 10**9  # ⭐(b) 탐색 범위 설정

while low < high:
    mid = (low + high + 1) // 2  # ⭐(c) 상향 이분탐색: +1 필수!
    if test(mid):
        low = mid   # ⭐(d) 가능하면 더 큰 값 탐색
    else:
        high = mid - 1  # ⭐(e) 불가능하면 범위 축소

print(low)
```

### 🔴 출제 포인트
| 빈칸 | 정답 | 출제 포인트 |
|------|------|-------------|
| ⭐(a) | `(x - a[i]) * (x - a[i])` | **제곱 비용 계산** |
| ⭐(b) | `1, 2 * 10**9` | **탐색 범위 설정** |
| ⭐(c) | `(low + high + 1) // 2` | **상향 이분탐색 +1** |
| ⭐(d) | `mid` | **조건 만족시 low 갱신** |
| ⭐(e) | `mid - 1` | **조건 불만족시 high 갱신** |

> 🔴 **핵심 암기**: 최대값 찾기 = `(low + high + 1) // 2` + `low = mid`

---

## 18. 스택 정렬 (DP)

### 문제 설명
수열을 스택을 이용하여 오름차순으로 정렬할 수 있는지 판별하는 문제.

**스택 정렬이 불가능한 조건**: $i < j < k$ 인데 $a[k] < a[i] < a[j]$ 인 경우
- 세 위치 i, j, k가 순서대로 있을 때, 값이 "중간값 < 최대값 < 최소값" 패턴이면 정렬 불가능

**예시**: 수열 [2, 3, 1]에서
- i=0, j=1, k=2일 때: a[0]=2, a[1]=3, a[2]=1
- $a[k]=1 < a[i]=2 < a[j]=3$ 이므로 정렬 불가능

### 제약 조건
- $1 \leq N \leq 5000$

### 입력 형식
```
N
a₁ a₂ ... aₙ
```

### 출력 형식
```
정렬 불가능을 증명하는 (i, j, k) 순서쌍의 총 개수
```

### 풀이 전략
- 모든 (i, j, k) 쌍을 $O(n^3)$으로 확인하면 시간 초과
- k를 고정하고, `more[i][k]` = i와 k 사이에서 a[i]보다 큰 값의 개수를 미리 계산
- $a[i] > a[k]$일 때 `total += more[i][k-1]`

### 시간복잡도
$$O(N^2)$$

### 공간복잡도
$$O(N^2)$$ (개선 가능: 1차원 배열로 $O(N)$)

### 정답 코드
```python
n = int(input())
a = list(map(int, input().split()))

more = [[0 for _ in range(n)] for _ in range(n)]
total = 0

for i in range(n):
    for k in range(i + 1, n):
        if a[i] < a[k]:
            more[i][k] = more[i][k - 1] + 1  # ⭐ a[i] < a[k]면 j 후보 추가
        else:
            more[i][k] = more[i][k - 1]      # 🔴 그대로 유지
            # ⭐⭐ 핵심! a[i] > a[k]일 때만 카운트
            total += more[i][k - 1]  # 🔴 a[k] < a[i] < a[j] 조건 충족

print(total)
```

### 🔴 출제 포인트
| 위치 | 출제 예상 | 설명 |
|------|-----------|------|
| ⭐ `more[i][k - 1] + 1` | **DP 점화식** | a[i] < a[k]면 j 후보 1개 추가 |
| ⭐ `more[i][k - 1]` | **DP 점화식** | a[i] >= a[k]면 개수 유지 |
| ⭐⭐ `total += more[i][k - 1]` | **카운팅 조건** | a[i] > a[k]일 때만! |

> 🔴 **핵심 암기**: 231 패턴 = a[k] < a[i] < a[j] (값 기준으로 "작 < 중 < 큰"이 **아닌** 경우)

---

## 19. 조직도 업무 처리 (완전이진트리 시뮬레이션)

### 문제 설명
완전이진트리 형태의 조직도에서 업무가 처리되는 과정을 시뮬레이션한다.

- 조직도 높이: h (말단 직원은 $2^h$명)
- 각 말단 직원은 k개의 업무를 가짐
- **홀수 날**: 왼쪽 부하의 업무 처리
- **짝수 날**: 오른쪽 부하의 업무 처리
- 업무는 말단 → 부서장까지 올라가야 완료 (h+1일 소요)
- **r일 동안 부서장이 처리한 업무 번호의 합** 출력

### 제약 조건
- $1 \leq h \leq 10$
- $1 \leq k \leq 10^5$
- $1 \leq r \leq 10^9$

### 입력 형식
```
h k r
task₁ (k개의 정수)
task₂ (k개의 정수)
...
task_{2^h} (k개의 정수)
```

### 출력 형식
```
r일 동안 부서장이 처리한 업무 번호의 합
```

### 풀이 전략
- 트리를 시뮬레이션하지 않고, **역으로 추적**하여 부서장에게 도달하는 업무 순서를 미리 계산
- merge 함수로 두 부하의 업무 리스트를 "교대로" 합침
- 최종 결과에서 처음 (r-h)개의 합 출력

### 시간복잡도
$$O(h \times 2^h \times k)$$

### 정답 코드
```python
def merge(list1, list2):
    lst = []
    for i in range(len(list1)):
        lst.append(list1[i])  # ⭐ 번갈아 합치기
        lst.append(list2[i])  # ⭐ 번갈아 합치기
    return lst

h, k, r = map(int, input().split())
tasks = []
for _ in range(2**h):  # 🔴 말단 직원 수 = 2^h
    tasks.append(list(map(int, input().split())))

for i in range(1, h + 1):  # 🔴 레벨 1부터 h까지
    tasks2 = []
    for j in range(2**(h - i)):  # ⭐ 현재 레벨 노드 수 = 2^(h-i)
        if i % 2:  # ⭐⭐ 홀수 레벨: 오른쪽 먼저 (= 왼쪽이 먼저 처리됨)
            tasks2.append(merge(tasks[2*j + 1], tasks[2*j]))
        else:      # ⭐⭐ 짝수 레벨: 왼쪽 먼저
            tasks2.append(merge(tasks[2*j], tasks[2*j + 1]))
    tasks = tasks2

print(sum(tasks[0][:r - h]))  # ⭐⭐⭐ 핵심! r-h개만 합산
```

### 🔴 출제 포인트
| 위치 | 출제 예상 | 설명 |
|------|-----------|------|
| ⭐ `2**h` | **말단 직원 수** | 완전이진트리 |
| ⭐ `2**(h - i)` | **레벨별 노드 수** | |
| ⭐⭐ `i % 2` | **홀짝 분기** | 홀수/짝수 날 처리 순서 |
| ⭐⭐ `tasks[2*j + 1], tasks[2*j]` | **merge 순서** | 홀수 레벨 |
| ⭐⭐⭐ `r - h` | **완료 업무 개수** | 말단→부서장 h+1일 소요 |

> 🔴 **핵심 암기**: 
> - r일까지 완료된 업무 = **r - h**개 (1일차 시작 → h+1일차 완료)
> - 홀수 레벨 merge: `merge(right, left)` → 실제론 left 먼저 처리

---

## 20. 대회 등수 계산 (정렬)

### 문제 설명
n명의 사원이 3개의 대회에 참가한다. 각 대회별 등수와 최종(3개 합산) 등수를 출력한다.

**등수 규칙**: "나보다 점수가 높은 사람 수 + 1" (동점자는 같은 등수)
- 예: 점수가 [100, 80, 80, 70]이면 등수는 [1, 2, 2, 4]

### 제약 조건
- $1 \leq N \leq 10^5$

### 입력 형식
```
N
score₁₁ score₁₂ ... score₁ₙ  (1번 대회)
score₂₁ score₂₂ ... score₂ₙ  (2번 대회)
score₃₁ score₃₂ ... score₃ₙ  (3번 대회)
```

### 출력 형식
```
1번 대회 등수 (공백 구분)
2번 대회 등수 (공백 구분)
3번 대회 등수 (공백 구분)
최종 등수 (공백 구분)
```

### 풀이 전략
1. [점수, 원래 인덱스] 형태로 변환
2. 점수 기준 내림차순 정렬
3. 순서대로 등수 부여 (동점이면 같은 등수 유지)
4. 원래 인덱스 순서로 복원하여 출력

### 시간복잡도
$$O(N \log N)$$

### 정답 코드
```python
n = int(input())

def computeRanking(arr):
    ranking = [0] * n
    ans = [0] * n
    
    for i in range(n):
        arr[i] = [arr[i], i]  # ⭐ [점수, 원래인덱스] 변환
    
    arr.sort(reverse=True)  # 🔴 내림차순 정렬 (높은 점수 먼저)
    
    cnt = 1
    ranking[0] = 1  # 🔴 1등은 무조건 1
    
    for i in range(1, n):
        if arr[i][0] != arr[i-1][0]:  # ⭐⭐ 동점 아니면
            cnt = i + 1  # ⭐⭐ 등수 = 현재 순위 (1, 2, 2, 4 형태)
        ranking[i] = cnt
    
    for i in range(n):
        ans[arr[i][1]] = ranking[i]  # ⭐ 원래 인덱스로 복원
    
    for i in range(n):
        print(ans[i], end=' ')
    print()

total = [0] * n

for _ in range(3):
    scores = list(map(int, input().split()))
    for i in range(n):
        total[i] += scores[i]
    computeRanking(scores)

computeRanking(total)
```

### 🔴 출제 포인트
| 위치 | 출제 예상 | 설명 |
|------|-----------|------|
| ⭐ `[arr[i], i]` | **인덱스 보존** | 정렬 후 복원 위해 |
| 🔴 `reverse=True` | **내림차순** | 높은 점수 = 높은 등수 |
| ⭐⭐ `cnt = i + 1` | **동점 처리** | 공동 2등 다음은 4등 |
| ⭐ `ans[arr[i][1]]` | **원래 순서 복원** | |

> 🔴 **핵심 암기**: 동점이면 등수 유지, 동점 끝나면 **현재 인덱스+1**로 점프

---

## 21. 초염기서열 (비트마스크 DP)

### 문제 설명
n개의 "좋은 염기서열"이 주어진다. 각 염기서열은 길이 m이며, 와일드카드 '.'을 포함할 수 있다.

- **초염기서열**: 여러 좋은 염기서열을 동시에 만족하는 실제 염기서열
- 목표: **모든 좋은 염기서열을 커버하는 최소 초염기서열 개수**

**예시**:
- "a..tt"와 "a.g.t"는 "acgtt"라는 하나의 초염기서열로 커버 가능
- "a..."와 "g..."는 충돌 → 서로 다른 초염기서열 필요

### 제약 조건
- $1 \leq n \leq 15$
- $1 \leq m \leq 50$

### 입력 형식
```
n m
dna₁
dna₂
...
dnaₙ
```

### 출력 형식
```
모든 염기서열을 커버하는 최소 초염기서열 개수
```

### 풀이 전략
1. 비트마스크로 염기서열 집합 표현 ($2^n$개의 부분집합)
2. 각 부분집합에 대해 "하나의 초염기서열로 커버 가능한지" 미리 계산
3. DP로 전체 집합을 최소 개수의 "커버 가능한 부분집합"으로 분할

### 시간복잡도
$$O(3^n + 2^n \times m)$$

### 정답 코드
```python
n, m = map(int, input().split())
dna = [list(input()) for _ in range(n)]

superDNA = [None for _ in range(2**n)]  # 🔴 2^n개 부분집합
superDNA[0] = ['.'] * m  # ⭐ 공집합 = 모두 와일드카드

def merge(dna1, dna2):
    if dna1 == [] or dna2 == []:
        return []  # 🔴 이미 불가능한 경우
    result = []
    for i in range(m):
        if dna1[i] == '.':
            result.append(dna2[i])      # ⭐ '.' + X = X
        elif dna2[i] == '.':
            result.append(dna1[i])      # ⭐ X + '.' = X
        elif dna1[i] == dna2[i]:
            result.append(dna1[i])      # ⭐ X + X = X
        else:
            return []  # ⭐⭐ 충돌! 다른 문자 = 병합 불가
    return result

def genSuperDNA(index):
    loc = 0
    tempIndex = index
    while tempIndex % 2 == 0:  # ⭐⭐ 가장 오른쪽 1비트 찾기
        tempIndex //= 2
        loc += 1
    superDNA[index] = merge(dna[loc], superDNA[index - 2**loc])  # ⭐⭐ 증분적 계산

for i in range(1, 2**n):
    genSuperDNA(i)

answer = [n + 1] * (2**n)  # 🔴 불가능 = 큰 값
answer[0] = 0  # ⭐ 공집합 = 0개 필요

def genAnswer(index):
    if answer[index] < n + 1:
        return answer[index]  # 🔴 메모이제이션
    minVal = n + 1
    sub = index
    while sub > 0:
        other = index ^ sub  # ⭐⭐ XOR로 나머지 집합
        if other > 0:
            val = genAnswer(sub) + genAnswer(other)
            if val < minVal:
                minVal = val
        sub = (sub - 1) & index  # ⭐⭐⭐ 부분집합 열거 트릭!
    answer[index] = minVal
    return minVal

for i in range(1, 2**n):
    if superDNA[i] != []:
        answer[i] = 1  # 🔴 하나로 커버 가능 = 1
    else:
        genAnswer(i)

print(answer[2**n - 1])  # ⭐⭐ 전체 집합 = 2^n - 1
```

### 🔴 출제 포인트
| 위치 | 출제 예상 | 설명 |
|------|-----------|------|
| ⭐ merge 함수 전체 | **병합 규칙** | `.` 처리, 충돌 처리 |
| ⭐⭐ `tempIndex % 2 == 0` | **오른쪽 1비트 찾기** | |
| ⭐⭐ `index ^ sub` | **XOR 연산** | 나머지 집합 계산 |
| ⭐⭐⭐ `(sub - 1) & index` | **부분집합 열거** | 핵심 비트 트릭! |
| ⭐⭐ `2**n - 1` | **전체 집합** | 모든 비트 1 |

> 🔴 **핵심 암기**: 
> - 부분집합 열거: `sub = (sub - 1) & index`
> - 전체 집합 비트마스크: `2^n - 1` = `0b111...1` (n개의 1)

---

## 22. 출퇴근길 (그래프 DFS)

### 문제 설명
단방향 그래프에서 집(S) → 회사(T) 출근길과 회사(T) → 집(S) 퇴근길이 있다.

**출근길과 퇴근길 "모두"에서 방문 가능한 정점의 개수**를 구한다.
(단, S와 T 자체는 제외)

### 제약 조건
- $1 \leq N \leq 10^5$ (정점 개수)
- $1 \leq M \leq 2 \times 10^5$ (간선 개수)

### 입력 형식
```
N M
x₁ y₁
x₂ y₂
...
xₘ yₘ
S T
```

### 출력 형식
```
출퇴근길 모두에서 방문 가능한 정점의 개수 (S, T 제외)
```

### 풀이 전략
4가지 DFS로 각 조건 체크:
1. `fromS[v]`: S에서 v로 도달 가능? (T를 거치지 않고)
2. `fromT[v]`: T에서 v로 도달 가능? (S를 거치지 않고)
3. `RfromS[v]`: v에서 S로 도달 가능? (역방향 그래프)
4. `RfromT[v]`: v에서 T로 도달 가능? (역방향 그래프)

**4가지 모두 만족하면 출퇴근길 모두 방문 가능!**

### 시간복잡도
$$O(N + M)$$

### 정답 코드
```python
import sys
sys.setrecursionlimit(10**7)  # 🔴 재귀 제한 해제 필수!

n, m = map(int, input().split())

adj = [[] for _ in range(n + 1)]   # 정방향
adjR = [[] for _ in range(n + 1)]  # ⭐ 역방향 그래프

for _ in range(m):
    x, y = map(int, input().split())
    adj[x].append(y)   # 정방향: x → y
    adjR[y].append(x)  # ⭐⭐ 역방향: y → x (간선 뒤집기)

s, t = map(int, input().split())

def dfs(now, graph, visited):
    if visited[now] == 1:
        return
    visited[now] = 1
    for neighbor in graph[now]:
        dfs(neighbor, graph, visited)

# ⭐⭐⭐ 출근길: S→v (T 차단), v→T (역방향)
fromS = [0] * (n + 1)
fromS[t] = 1  # ⭐⭐ T를 미리 방문 처리 → T 이후 탐색 차단!
dfs(s, adj, fromS)

# ⭐⭐⭐ 퇴근길: T→v (S 차단), v→S (역방향)
fromT = [0] * (n + 1)
fromT[s] = 1  # ⭐⭐ S를 미리 방문 처리 → S 이후 탐색 차단!
dfs(t, adj, fromT)

# ⭐ 역방향 DFS: v→S 가능 여부
RfromS = [0] * (n + 1)
dfs(s, adjR, RfromS)  # 🔴 역방향 그래프 사용!

# ⭐ 역방향 DFS: v→T 가능 여부
RfromT = [0] * (n + 1)
dfs(t, adjR, RfromT)  # 🔴 역방향 그래프 사용!

count = 0
for i in range(1, n + 1):
    # ⭐⭐⭐ 4가지 조건 모두 만족
    if fromS[i] == 1 and fromT[i] == 1 and RfromS[i] == 1 and RfromT[i] == 1:
        count += 1

print(count - 2)  # ⭐⭐ S, T 제외하므로 -2
```

### 🔴 출제 포인트
| 위치 | 출제 예상 | 설명 |
|------|-----------|------|
| ⭐⭐ `adjR[y].append(x)` | **역방향 그래프** | 간선 방향 뒤집기 |
| ⭐⭐ `fromS[t] = 1` | **T 차단** | T 도착 후 더 이상 안 감 |
| ⭐⭐ `fromT[s] = 1` | **S 차단** | S 도착 후 더 이상 안 감 |
| 🔴 `dfs(s, adjR, ...)` | **역방향 DFS** | v→S 가능 여부 |
| ⭐⭐ 4가지 AND 조건 | **최종 판별** | |
| ⭐⭐ `count - 2` | **S, T 제외** | |

> 🔴 **핵심 암기**: 
> - "v에서 X로 도달 가능" = **역방향 그래프에서 X부터 DFS**
> - T 차단: `fromS[t] = 1` (미리 방문 처리)
> - 최종 답: **4가지 조건 AND** 후 **-2** (S, T 제외)

---

## 23. 자동차 연비 중앙값 (이진 탐색)

### 문제 설명
n대의 자동차 중 3대를 골라 테스트한다. **3대의 연비 중앙값이 m이 되는 경우의 수**를 구한다.
q개의 쿼리에 대해 각각 답을 출력한다.

### 제약 조건
- $1 \leq N, Q \leq 2 \times 10^5$

### 입력 형식
```
N Q
mileage₁ mileage₂ ... mileageₙ
m₁
m₂
...
mq
```

### 출력 형식
```
각 쿼리에 대해 경우의 수 출력
```

### 풀이 전략
3대를 골라 중앙값이 m이 되려면:
- 1대는 m보다 작은 연비
- 1대는 정확히 m
- 1대는 m보다 큰 연비

**경우의 수 = (m보다 작은 수) × (m보다 큰 수)**

### 시간복잡도
$$O(N \log N + Q \log N)$$

### 정답 코드
```python
import bisect

n, q = map(int, input().split())
mileage = list(map(int, input().split()))
mileage.sort()  # 🔴 정렬 필수!

for _ in range(q):
    m = int(input())
    idx = bisect.bisect_left(mileage, m)  # ⭐⭐ m의 왼쪽 경계 = m보다 작은 개수
    
    # ⭐⭐⭐ m이 실제로 존재하는지 확인 (매우 중요!)
    if idx != n and m == mileage[idx]:
        # ⭐ idx = m보다 작은 원소 개수
        # ⭐ (n - idx - 1) = m보다 큰 원소 개수 (-1은 m 자신 제외)
        print(idx * (n - idx - 1))  # ⭐⭐⭐ 핵심 공식!
    else:
        print(0)  # 🔴 m이 없으면 불가능
```

### 🔴 출제 포인트
| 위치 | 출제 예상 | 설명 |
|------|-----------|------|
| 🔴 `mileage.sort()` | **정렬** | bisect 사용 전제 |
| ⭐⭐ `bisect.bisect_left` | **왼쪽 경계** | m보다 작은 개수 |
| ⭐⭐⭐ `idx != n and m == mileage[idx]` | **존재 여부 확인** | 핵심 조건! |
| ⭐⭐⭐ `idx * (n - idx - 1)` | **경우의 수 공식** | (작은 수) × (큰 수) |

> 🔴 **핵심 암기**: 
> - `bisect_left(arr, x)` = x보다 **작은** 원소 개수
> - 중앙값 경우의 수 = **(작은 것 개수) × (큰 것 개수)**
> - m 존재 확인: `idx != n and arr[idx] == m`

---

## 24. 격자 경로 탐색 (DFS 백트래킹)

### 문제 설명
n×n 격자에서 m개의 지점을 **순서대로** 방문한다.
- 상하좌우 이동만 가능
- 벽(1) 통과 불가
- 한 번 지나간 칸은 다시 방문 불가
- **가능한 경로의 수** 출력

### 제약 조건
- $1 \leq N \leq 10$
- $2 \leq M \leq 10$

### 입력 형식
```
N M
grid (N×N, 0: 빈 칸, 1: 벽)
dest₁_x dest₁_y
dest₂_x dest₂_y
...
destₘ_x destₘ_y
```

### 출력 형식
```
가능한 경로의 수
```

### 풀이 전략
- **DFS + 백트래킹**으로 모든 경로 탐색
- 현재 목적지(destIdx)에 도착하면 다음 목적지로 변경
- 마지막 목적지 도착 시 cnt 증가
- 방문 표시 후 탐색, 탐색 완료 후 방문 해제 (백트래킹)

### 시간복잡도
$$O(4^{N^2})$$ 최악, 실제로는 백트래킹으로 크게 감소

### 정답 코드
```python
def dfs(now, destIdx):
    global cnt
    
    # ⭐ 현재 위치가 목적지인지 확인
    if now == dest[destIdx]:
        if destIdx == m - 1:  # ⭐⭐ 마지막 목적지 도착!
            cnt += 1
            return
        else:
            destIdx += 1  # ⭐ 다음 목적지로 전환
    
    x, y = now
    visit[x][y] = True  # ⭐⭐ 방문 표시
    
    for i in range(4):
        nx, ny = x + dx[i], y + dy[i]
        # ⭐ 범위 체크 & 미방문 & 빈 칸
        if 0 <= nx < n and 0 <= ny < n and \
           not visit[nx][ny] and grid[nx][ny] == 0:
            dfs([nx, ny], destIdx)
    
    visit[x][y] = False  # ⭐⭐⭐ 백트래킹: 방문 해제!

n, m = map(int, input().split())
grid = [list(map(int, input().split())) for _ in range(n)]

dest = []
for _ in range(m):
    x, y = map(int, input().split())
    dest.append([x - 1, y - 1])  # ⭐ 1-indexed → 0-indexed

visit = [[False] * n for _ in range(n)]
cnt = 0
dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]  # 🔴 상하좌우

dfs(dest[0], 1)  # ⭐⭐ 첫 목적지에서 시작, destIdx=1부터
print(cnt)
```

### 🔴 출제 포인트
| 위치 | 출제 예상 | 설명 |
|------|-----------|------|
| ⭐⭐ `destIdx == m - 1` | **종료 조건** | 마지막 목적지 |
| ⭐ `destIdx += 1` | **목적지 전환** | |
| ⭐⭐ `visit[x][y] = True` | **방문 표시** | DFS 진입 시 |
| ⭐⭐⭐ `visit[x][y] = False` | **백트래킹** | DFS 탈출 시 해제! |
| ⭐ `[x - 1, y - 1]` | **좌표 변환** | 1-indexed → 0-indexed |
| ⭐⭐ `dfs(dest[0], 1)` | **시작 조건** | 첫 목적지에서, 두 번째 목적지 향해 |

> 🔴 **핵심 암기**: 
> - 백트래킹 = **visit 해제** (`visit[x][y] = False`)
> - 시작: `dfs(dest[0], 1)` → 첫 목적지에서 시작, destIdx=**1**부터
> - 종료: `destIdx == m - 1` (0-indexed이므로)

---

## 📝 전체 요약: 출제 예상 빈칸 TOP 10

| 순위 | 문제 | 빈칸 내용 | 난이도 |
|------|------|-----------|--------|
| 1 | **17** | `(low + high + 1) // 2` | ⭐⭐⭐ |
| 2 | **17** | `(x - a[i]) * (x - a[i])` | ⭐⭐ |
| 3 | **22** | `adjR[y].append(x)` (역방향) | ⭐⭐⭐ |
| 4 | **22** | `fromS[t] = 1` (T 차단) | ⭐⭐⭐ |
| 5 | **23** | `idx * (n - idx - 1)` | ⭐⭐⭐ |
| 6 | **24** | `visit[x][y] = False` (백트래킹) | ⭐⭐⭐ |
| 7 | **21** | `(sub - 1) & index` (부분집합 열거) | ⭐⭐⭐ |
| 8 | **19** | `r - h` (완료 업무 개수) | ⭐⭐ |
| 9 | **18** | `total += more[i][k - 1]` | ⭐⭐ |
| 10 | **20** | `cnt = i + 1` (동점 처리) | ⭐⭐ |

---

## 🔥 암기 필수 공식

```python
# 1. 상향 이분탐색 (최대값 찾기)
mid = (low + high + 1) // 2
if 조건: low = mid
else: high = mid - 1

# 2. 하향 이분탐색 (최소값 찾기)
mid = (low + high) // 2
if 조건: high = mid
else: low = mid + 1

# 3. 부분집합 열거
sub = index
while sub > 0:
    # sub 처리
    sub = (sub - 1) & index

# 4. 역방향 그래프로 "v→X 도달 가능" 확인
dfs(X, adjR, visited)  # 역방향에서 X부터 DFS

# 5. 백트래킹
visit[x][y] = True   # 진입
dfs(...)
visit[x][y] = False  # 탈출 시 해제

# 6. 중앙값 경우의 수
bisect_left(arr, m) * (n - bisect_left(arr, m) - 1)
```
