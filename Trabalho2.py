#Trabalho 2 Funções Interpolação Polinomial e Ajuste de Curvas
import numpy as np
#Instalar o app scipy
#pip install scipy
from scipy.linalg import lu, cholesky

def grau_metodo():
    grau = int(input("Digite o grau do polinômio desejado: "))
    metodo = input("Digite o método desejado (LU ou Cholesky): ")
    return grau, metodo

def entrada_dados(interpolacao=True):
    try:
        n = int(input("Digite o número de pontos tabelados: "))
    except ValueError:
        print("Erro: número de pontos tabelados deve ser um número inteiro.")
        return

    tabela = []
    for i in range(n):
        try:
            x = float(input(f"Digite o valor de x_{i+1}: "))
            y = float(input(f"Digite o valor de y_{i+1}: "))
        except ValueError:
            print("Erro: os valores de x e y devem ser números.")
            return
        tabela.append((x, y))

    if(interpolacao):
        try:
            x_interpolacao = float(input("Digite o ponto onde se deseja conhecer o P(x) interpolado: "))
        except ValueError:
            print("Erro: o ponto de interpolação deve ser um número.")
            return
        return n, tabela, x_interpolacao

    return n, tabela

def interpolacao_newton(n, tabela, x):
    diferencas = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        diferencas[i][0] = tabela[i][1]
    
    # Calcula as diferenças divididas
    for j in range(1, n):
        for i in range(n - j):
            diferencas[i][j] = (diferencas[i+1][j-1] - diferencas[i][j-1]) / (tabela[i+j][0] - tabela[i][0])
    
    interpolado = diferencas[0][0]
    termo = 1.0
    for i in range(1, n):
        termo *= (x - tabela[i-1][0])
        interpolado += diferencas[0][i] * termo
    
    return interpolado

def interpolacao_newton_gregory(n, tabela, x):
    # Inicializa a matriz de diferenças divididas
    diferencas = [[0 for _ in range(n)] for _ in range(n)]
    
    # Preenche a primeira coluna com os valores de y da tabela
    for i in range(n):
        diferencas[i][0] = tabela[i][1]
    
    # Calcula as diferenças divididas
    for j in range(1, n):
        for i in range(n - j):
            diferencas[i][j] = diferencas[i+1][j-1] - diferencas[i][j-1]
    
    # Calcula o valor interpolado
    interpolado = diferencas[0][0]
    termo = 1.0
    for i in range(1, n):
        termo *= (x - tabela[i-1][0])
        interpolado += diferencas[0][i] * termo
    
    return interpolado

def coeficiente_determinacao(n, tabela):
    y_ajustados = []
    for i in range(n):
        y_ajustado = interpolacao_newton_gregory(n, tabela, tabela[i][0])
        y_ajustados.append(y_ajustado)
    
    y_observados = [tabela[i][1] for i in range(n)]
    y_medio = sum(y_observados) / n
    
    ss_total = sum((y - y_medio)**2 for y in y_observados)
    ss_res = sum((y_observado - y_ajustado)**2 for y_observado, y_ajustado in zip(y_observados, y_ajustados))
    
    r2 = 1 - (ss_res / ss_total)
    
    return y_ajustados, r2

def ajuste_reta_minimos_quadrados(n, tabela):
    # Pega os valores da tabela
    x = np.array([t[0] for t in tabela])
    y = np.array([t[1] for t in tabela])
    
    # Calcula os coeficientes a0 e a1 da reta
    a1, a0 = np.polyfit(x, y, 1)
    
    # Calcula os valores Y ajustados
    y_ajustados = a0 + a1 * x
    
    # Calcula o coeficiente de determinação R^2 usando a função coeficiente_determinacao
    _, r2 = coeficiente_determinacao(n, [(xi, yi) for xi, yi in zip(x, y_ajustados)])
    
    return a0, a1, y_ajustados, r2

def ajuste_polinomial(n, tabela, grau, metodo):
    x = np.array([t[0] for t in tabela])
    y = np.array([t[1] for t in tabela])
    
    A = np.vander(x, grau + 1, increasing=True)
    
    if metodo.upper() == 'LU':
        # Decomposição LU
        P, L, U = lu(A)
        coeficientes = np.linalg.solve(U, np.linalg.solve(L, np.dot(P, y)))
    elif metodo.upper() == 'CHOLESKY':
        # Decomposição de Cholesky
        L = cholesky(np.dot(A.T, A))
        coeficientes = np.linalg.solve(L.T, np.linalg.solve(L, np.dot(A.T, y)))
    else:
        raise ValueError("Método não reconhecido. Escolha entre 'LU' e 'Cholesky'.")
    
    # Calcula os valores Y ajustados
    y_ajustados = np.dot(A, coeficientes)
    
    # Calcula o coeficiente de determinação R^2
    r2 = coeficiente_determinacao(y, y_ajustados)
    
    return coeficientes, y_ajustados, r2

def ajuste_curva_exponencial(n, tabela):
    x = np.array([t[0] for t in tabela])
    y = np.array([t[1] for t in tabela])
    
    # Aplica a transformação para linearizar
    y_log = np.log(y)
    x_log = x
    
    # Calcula os coeficientes a e ln(b) da reta
    A = np.vstack([x_log, np.ones(len(x_log))]).T
    a, ln_b = np.linalg.lstsq(A, y_log, rcond=None)[0]
    
    b = np.exp(ln_b)
    
    # Calcula os valores Y ajustados
    y_ajustados = a * b**x
    
    # Calcula o coeficiente de determinação R^2
    r2 = coeficiente_determinacao(y, y_ajustados)
    
    return a, b, y_ajustados, r2

# Menu para escolher a função
def menu():
    try:
        print("Escolha uma das funções:")
        print("0. Sair")
        print("1. Interpolação Newton")
        print("2. Interpolação de Newton-Gregory")
        print("3. Retorna o coeficiente de determinação entre os pontos tabelados e os pontos ajustados.")
        print("4. Ajusta os pontos tabelados a uma reta da forma y = a0 + a1x")
        print("5. Ajusta os pontos tabelados a um polinômio de grau desejado")
        print("6. Ajusta os pontos tabelados a uma curva exponencial da forma y = abx")

        opcao = int(input("Digite o número da função desejada: "))
        return opcao
    except Exception:
        raise Exception("Erro ao escolher a função. Verifique o valor digitado...")

while True:
    opcao = menu()
    
    if(opcao == 1 or opcao == 2):
        N,tabela,X = entrada_dados(True)
    if opcao == 0:
        print("Programa encerrado.")
        break
    if opcao == 1:
        try:
            N,tabela,X = entrada_dados(True)
            I = interpolacao_newton(N,tabela,X)
            print("Interpolado: ", I)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 2:
        try:
            N,tabela,X = entrada_dados(True)
            I = interpolacao_newton_gregory(N,tabela,X)
            print("Interpolado: ", I)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 3:
        try:
            N,tabela = entrada_dados(False)
            V, R = coeficiente_determinacao(N, tabela)
            print("Vetor: ", V)
            print("Real: ", R)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 4:
        try:
            N,tabela = entrada_dados(False)
            a0,a1,ajust,R = ajuste_reta_minimos_quadrados(N,tabela)
            print("A0: ", a0)
            print("A1: ", a1)
            print("Y Ajustado: ", ajust)
            print("Real: ", R)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 5:
        try:
            N,tabela = entrada_dados(False)
            grau, metodo = grau_metodo()
            V, ajust, R = ajuste_polinomial(N, tabela, grau, metodo)
            print("Coeficientes: ", V)
            print("Y Ajustado: ", ajust)
            print("Real: ", R)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    elif opcao == 6:
        try:
            N,tabela = entrada_dados(False)
            a, b, ajust, R = ajuste_curva_exponencial(N, tabela)
            print("1º Coeficiente: ", a)
            print("2º Coeficiente: ", b)
            print("Y Ajustado: ", ajust)
            print("Real: ", R)
        except Exception as e: 
            print(e)
        input("Pressione Enter para continuar...")
    else:
        input("Função não encontrada. Pressione enter para digitar novamente...")
        
    