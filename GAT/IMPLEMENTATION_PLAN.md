# Implementation Plan: GAT + Distributed Optimization for Multi-Robot MICP

> **Amaç:** Makaleyi adim adim anlayarak, küçük parçalar halinde implement etmek.
> **Kaynak:** Le et al., "Combining Graph Attention Networks and Distributed Optimization for Multi-Robot Mixed-Integer Convex Programming" (arXiv:2503.21548v1)

---

## Faz 0: Temel Kavramları Anlama

Bu fazda kod yazmıyoruz. Sadece problem uzayını ve kavramları anlıyoruz.

- [x] **0.1** Problem nedir? Multi-robot navigation with obstacle avoidance
- [x] **0.2** Neden "Mixed-Integer"? Çarpışma engelleme kısıtları OR içeriyor → big-M ile binary değişkenlere dönüşüyor
- [x] **0.3** Neden "Convex"? Binary değişkenler sabitlenince kalan problem quadratic cost + linear constraints = convex
- [x] **0.4** Neden Graph Attention Network? Robotlar ve engeller arası ilişkiler doğal olarak bir graf oluşturuyor, GAT bu yapıyı öğrenebiliyor
- [x] **0.5** Framework'ün büyük resmi: Offline (GAT eğit) → Online (GAT ile binary tahmin et → convex problemi çöz)
- **Notlar:** `plots/faz0_kavramlar.md` dosyasında detaylı açıklamalar mevcut.

---

## Faz 1: 2D Ortam ve Robot Dinamikleri

Tek bir robotu hareket ettirebildiğimiz basit bir simülasyon ortamı.

- [x] **1.1** 2D ortam oluştur: sınırlar (px_min, px_max, py_min, py_max), dikdörtgen engeller
- [x] **1.2** Double-integrator dinamik modeli implement et: `p(k+1) = p(k) + τv(k) + 0.5τ²u(k)`, `v(k+1) = v(k) + τu(k)`
- [x] **1.3** Tek robot, engelsiz: başlangıçtan hedefe basit trajectory üret (QP ile veya elle)
- [x] **1.4** Görselleştirme: matplotlib ile robotun hareketini, engelleri ve hedefi çiz
- [x] **1.5** Bound constraints ekle: hız limiti, ivme limiti, ortam sınırları

**Bu fazın sonunda anlaşılacak:** Robotun dinamik modeli, state-space temsili, basit trajectory planning

---

## Faz 2: Tek Robot + Engel ile MICP

Tek bir robotun tek bir engelden kaçınma problemini big-M formülasyonu ile çözme.

- [x] **2.1** Big-M formülasyonunu anla: `|px_i - px_o| >= d OR |py_i - py_o| >= d` nasıl binary değişkenlere dönüşüyor?
- [x] **2.2** Robot-engel çarpışma kısıtlarını implement et (Denklem 6)
- [x] **2.3** Amaç fonksiyonunu implement et: hedefe yaklaş + minimum enerji (Denklem 7-8)
- [x] **2.4** GUROBI ile tek robot + tek engel MICP'yi çöz
- [x] **2.5** Çözümü görselleştir: robotun engelden kaçarak hedefe giden trajectory'si
- [x] **2.6** Receding horizon (MPC) loop: her adımda MICP çöz, ilk control input'u uygula, tekrarla

**Bu fazın sonunda anlaşılacak:** Big-M formülasyonu, MICP yapısı, binary değişkenlerin rolü, receding horizon mantığı

---

## Faz 3: Çoklu Robot MICP

Birden fazla robotun birbirleriyle ve engellerle çarpışmadan navigasyon yapması.

- [x] **3.1** Robot-robot çarpışma kısıtlarını implement et (Denklem 4): big-M ile inter-robot collision avoidance
- [x] **3.2** Proximity-based edge oluşturma: sadece yakın robotlar arasında kısıt koy (Denklem 5, dprox eşiği)
- [x] **3.3** 2 robot + 1 engel senaryosu çöz ve görselleştir
- [x] **3.4** 3 robot + 2 engel senaryosu çöz ve görselleştir
- [x] **3.5** Robot sayısı arttıkça GUROBI çözüm süresini ölç ve kaydet
- [x] **3.6** Heterogeneous graph yapısını anla: V = R ∪ O, E = ER ∪ ERO ∪ EOR ∪ EO

**Bu fazın sonunda anlaşılacak:** Multi-agent MICP karmaşıklığı, coupling constraints, graph yapısının doğal olarak ortaya çıkışı

---

## Faz 4: Veri Üretimi

GAT eğitimi için dataset oluşturma.

- [x] **4.1** Rastgele senaryo üretici yaz: N robot, M engel, rastgele başlangıç/hedef pozisyonları
- [x] **4.2** Her senaryo için GUROBI ile MICP çöz, binary çözümleri topla
- [x] **4.3** Binary çözüm refinement: ill-posed durumu düzelt (bir constraint zaten sağlanıyorsa binary'yi 0 yap)
- [x] **4.4** Dataset formatı: (graph, node features, edge binary labels)
- [x] **4.5** 2-5 robot ile yeterli sayıda veri üret (~5000-10000 başlangıç için yeterli)
- [x] **4.6** Train/validation split (%90/%10)

**Bu fazın sonunda anlaşılacak:** Supervised learning için veri üretim süreci, parametric MICP kavramı, veri kalitesi önemi

---

## Faz 5: Graph Attention Network (GAT) Eğitimi

Heterogeneous GAT network tasarlama ve eğitme.

- [x] **5.1** PyTorch Geometric kurulumu ve temel kavramlar: node features, edge index, heterogeneous graph
- [x] **5.2** Projection layer: farklı node tiplerini (robot vs engel) aynı feature space'e taşı
- [x] **5.3** GAT layer'ı anla: attention mekanizması nasıl çalışıyor? (Denklem 11-12)
- [x] **5.4** Encoder implement et: Projection → 2-layer GAT (64 nöron)
- [x] **5.5** Decoder implement et: edge embedding (iki node'un embedding'ini birleştir) → feedforward NN → binary prediction
- [x] **5.6** Ayrı decoder'lar: robot-robot kenarları için ΩR, robot-engel kenarları için ΩRO
- [ ] **5.7** Cross-entropy loss ile eğitim
- [ ] **5.8** Validation accuracy ölç (hedef: ~%90+ robot-engel, ~%90+ robot-robot)

**Bu fazın sonunda anlaşılacak:** GAT'in neden bu probleme uygun olduğu, heterogeneous graph learning, attention mekanizması

---

## Faz 6: Online Pipeline — GAT + Convex Solver

Eğitilmiş GAT'i kullanarak online trajectory planning.

- [ ] **6.1** GAT ile binary tahmin et → kalan convex QP'yi GUROBI ile çöz
- [ ] **6.2** Infeasibility handling: soft constraints / slack variables ile çöz
- [ ] **6.3** Tam receding horizon loop: her adımda GAT predict → QP solve → uygula → tekrarla
- [ ] **6.4** Karşılaştırma: GAT+QP vs tam MICP (GUROBI) — çözüm kalitesi ve süre
- [ ] **6.5** Başarı oranı ve çarpışma oranı metrikleri hesapla
- [ ] **6.6** Farklı robot sayılarıyla test et (eğitimde görülmemiş sayılar dahil)

**Bu fazın sonunda anlaşılacak:** Framework'ün uçtan uca çalışması, ML+optimization hibrit yaklaşımın avantajları

---

## Faz 7 (Opsiyonel): Distributed ADMM

Bu faz opsiyonel — ana fikri anlamak için şart değil ama ilgi duyarsan ekleyebiliriz.

- [ ] **7.1** ADMM temel kavramı: augmented Lagrangian, primal-dual updates
- [ ] **7.2** Proximal ADMM implement et (Algorithm 1)
- [ ] **7.3** Centralized QP vs Distributed ADMM karşılaştırması

---

## Notlar

- **Araçlar:** Python, NumPy, Matplotlib, GUROBI (gurobipy), PyTorch, PyTorch Geometric
- **GUROBI Lisans:** WLS Compute Server (90 gün)
- **Öncelik:** Anlama > Hız. Her fazda ne öğrendiğimizi not edelim.
- **Basitleştirmeler:** Makaledeki bazı detayları (unicycle tracking, EKF vb.) atlayabiliriz — bunlar fiziksel robot için gerekli, simülasyonda double-integrator yeterli.
