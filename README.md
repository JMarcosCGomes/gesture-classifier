# gesture-classifier

Projeto desenvolvido para a disciplina de Redes Neurais (2026.1) na UFRJ. O objetivo central não é apenas treinar um modelo para classificar gestos, mas sim caracterizar a complexidade do problema através de variações sistemáticas de arquitetura e representação de dados.


## Instalação

### Clone o repositório
```bash
git clone git@github.com:JMarcosCGomes/gesture-classifier.git
```

### Crie e ative uma venv
```bash
python3 -m venv venv
```
```bash
source venv/bin/activate
```

### Instale as bibliotecas
```bash
pip install -r requirements.txt
```

## DATASET
O dataset utilizdo é o HaGRID 30k-384p, pode ser baixado pelo kaggle:
[https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p](https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p)

Após baixar coloque ele no diretório /data/raw/


## Execução dos scripts
os códigos foram feitos para serem rodados pelo terminal, escreva dessa forma no terminal (em `/gesture-classfier/` )
```bash
python3 -m src.codigoespecifico
```



## SETUP das representações (.csv)

Caso você não tenha ainda os `raw_landmarks.csv`, faça a extração com o seguinte código
```bash
python3 -m src.extract_raw
```

Após isso pode selecionar qual `process_*.py` rodar, é o mesmo processo

---

Para realizar os experimentos pode tanto ir pelo terminal diretamente 

```bash
python3 -m src.train --dataset landmarksreduced --model onelayer --train-frac 0.5
```



ou 


Você pode selecionar as opções desejadas no script `/scripts/run_experiments.sh` e rodar ele

```bash
chmod +x ./scripts/run_experiments.sh
./scripts/run_experiments.sh
```

Nele você pode selecionar as representações, arquiteturas e quantidade de amostras a serem experimentadas.



---
# Visualização

Também tenho dois notebooks que podem ser utilizados para visualizar o projeto.
- `experiment_analysis.ipynb`: Pode ser utilizado para analisar F1, accuracy, curva de treino e matriz de confusão dos experimentos.
- `features_analysis.ipynb`: Pode ser utilizado para visualizar o PCA dos ângulos e/ou distâncias.

