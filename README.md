# Urban Disaster Monitor

**Detecção de Civis e Socorristas em Imagens de Desastres Urbanos utilizando Visão Computacional**

## 📌 Introdução

Desastres urbanos, como desabamentos e enchentes, requerem respostas rápidas e eficazes por parte das equipes de resgate. O reconhecimento automático de civis e socorristas em imagens capturadas por drones, câmeras de segurança ou dispositivos móveis pode agilizar tomadas de decisão e alocação de recursos emergenciais.

Este projeto visa o desenvolvimento de um sistema inteligente baseado em visão computacional para **detectar e classificar indivíduos como civis ou socorristas em cenas de desastre urbano**. Utilizando técnicas de **detecção de objetos** com a arquitetura **YOLOv8**, buscamos construir uma ferramenta que possa auxiliar autoridades e equipes de resgate em tempo real.

## ⚙️ Técnicas e Metodologia

### Coleta e Anotação de Dados

- **Fonte do Dataset**: Imagens diversas de desastres urbanos (desabamentos, enchentes, etc.) obtidas de repositórios públicos e imagens sintéticas.
- **Anotação**: A anotação dos dados está sendo realizada na plataforma [Roboflow](https://roboflow.com), onde as classes principais são:
  - `People` - Civil
  - `Rescuer` - Socorrista

### Pré-processamento

- Redimensionamento das imagens para formatos compatíveis com o YOLOv8
- Aumento de dados (data augmentation) para balancear as classes e aumentar a robustez do modelo
- Divisão em conjunto de treino, validação e teste (ex: 70/20/10)

### Arquitetura Utilizada

- **Modelo**: YOLOv8 (You Only Look Once - versão 8)
- **Tarefa**: Detecção de Objetos (Object Detection)
- **Frameworks**: Ultralytics YOLOv8, PyTorch, OpenCV

### Etapas de Treinamento

1. Conversão das anotações da Roboflow para o formato YOLO
2. Treinamento do modelo com parâmetros ajustados (batch size, learning rate, número de epochs, etc.)
3. Validação com métricas de desempenho específicas
4. Testes com imagens reais e novos cenários

## 📊 Resultados Esperados

Os principais resultados a serem analisados incluem:

- **mAP@0.5** e **mAP@0.5:0.95** (Mean Average Precision)
- **Precisão e Recall** por classe
- **Confusion Matrix** para avaliação de erros entre classes (falsos positivos/negativos)
- **Testes qualitativos** com visualização das bounding boxes nas imagens

Estes resultados serão apresentados com gráficos e tabelas no relatório final, juntamente com uma análise crítica sobre a eficácia do modelo.

## 🔍 Conclusões (a ser preenchido após experimentos)

_(Espaço reservado para discussão sobre os resultados, limitações do modelo, melhorias possíveis e sugestões para aplicações reais.)_

---

## 💻 Execução Local

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/urban-disaster-monitor.git
cd urban-disaster-monitor

# Instalar dependências (requer Python 3.8+ e pip)
pip install -r requirements.txt

# Treinar modelo (exemplo com Ultralytics CLI)
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```

## 🧠 Repositório de Dados

O dataset anotado será disponibilizado (ou referenciado) no repositório assim que finalizado:

- Link Roboflow: _(a ser adicionado)_
- Formato: YOLOv8 (TXT + imagens)

## 📁 Estrutura do Projeto

```
EM CONSTRUÇÃO
```

## 📚 Referências

- Ultralytics YOLOv8 Docs: https://docs.ultralytics.com
- Roboflow: https://roboflow.com
- Redmon, J. et al. "You Only Look Once: Unified, Real-Time Object Detection"
- Dataset públicos: [Google Open Images](https://storage.googleapis.com/openimages/web/index.html), [Kaggle Datasets](https://www.kaggle.com/datasets)
