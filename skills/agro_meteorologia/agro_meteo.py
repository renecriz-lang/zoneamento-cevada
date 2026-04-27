#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SKILL: Agrometeorologia com Decêndios (ZARC-Ready)
Processa dados meteorológicos em decêndios com IDW + média climatológica
"""

import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

class AgroMeteorologia:
    def __init__(self, pasta_trabalho):
        self.pasta_trabalho = Path(pasta_trabalho)
        self.pasta_trabalho.mkdir(parents=True, exist_ok=True)
        
    def perguntar_parametros(self):
        """Interage com o usuário para definir parâmetros"""
        print("\n" + "="*70)
        print("AGROMETEOROLOGIA COM DECÊNDIOS - CONFIGURAÇÃO")
        print("="*70)
        
        # Pergunta 1: Intervalo
        print("\n1. Qual é o intervalo que quer usar?")
        print("   a) Decêndio (10 dias) - RECOMENDADO PARA ZARC")
        print("   b) Semana (7 dias)")
        print("   c) Pentada (5 dias)")
        print("   d) Mês completo")
        intervalo_input = input("Escolha (a/b/c/d) [padrão: a]: ").strip().lower() or "a"
        
        intervalo_map = {"a": "decendio", "b": "semana", "c": "pentada", "d": "mes"}
        self.intervalo = intervalo_map.get(intervalo_input, "decendio")
        
        # Pergunta 2: Quantas estações vizinhas
        print(f"\n2. Quantas estações vizinhas usar no IDW?")
        print(f"   (padrão: 3, aceita 3-15)")
        n_vizinhos = input("Escolha um número [padrão: 3]: ").strip() or "3"
        try:
            self.n_vizinhos = int(n_vizinhos)
            assert 3 <= self.n_vizinhos <= 15
        except:
            self.n_vizinhos = 3
            print(f"   → Usando 3 vizinhos")
        
        # Pergunta 3: Qual variável
        print(f"\n3. Qual variável processar?")
        print("   a) Precipitação")
        print("   b) Temperatura Média")
        print("   c) Temperatura Máxima")
        print("   d) Temperatura Mínima")
        print("   e) TODAS")
        var_input = input("Escolha (a/b/c/d/e) [padrão: a]: ").strip().lower() or "a"
        
        var_map = {
            "a": ["precipitacao"],
            "b": ["temperatura_media"],
            "c": ["temperatura_max"],
            "d": ["temperatura_min"],
            "e": ["precipitacao", "temperatura_media", "temperatura_max", "temperatura_min"]
        }
        self.variaveis = var_map.get(var_input, ["precipitacao"])
        
        print(f"\n✓ Parâmetros definidos:")
        print(f"  - Intervalo: {self.intervalo}")
        print(f"  - Vizinhos no IDW: {self.n_vizinhos}")
        print(f"  - Variáveis: {', '.join(self.variaveis)}")
        
    def carregar_estacoes(self, pasta_manuais, pasta_automaticas):
        """Carrega CSVs de estações manuais e automáticas"""
        print("\n" + "="*70)
        print("CARREGANDO ESTAÇÕES")
        print("="*70)
        
        estacoes = []
        
        # Manuais
        if Path(pasta_manuais).exists():
            csvs_manuais = glob.glob(str(Path(pasta_manuais) / "*.csv"))
            print(f"  Estações manuais encontradas: {len(csvs_manuais)}")
            for csv in csvs_manuais[:5]:  # Exemplo com 5 primeiras
                try:
                    df = pd.read_csv(csv, encoding='latin-1', sep=';', decimal=',')
                    # Extrai metadados do cabeçalho
                    codigo = "DESCONHECIDO"
                    nome = Path(csv).stem
                    tipo = "Manual"
                    estacoes.append({
                        "arquivo": csv,
                        "codigo": codigo,
                        "nome": nome,
                        "tipo": tipo,
                        "dados": df
                    })
                except Exception as e:
                    print(f"    ⚠ Erro ao ler {csv}: {e}")
        
        # Automáticas
        if Path(pasta_automaticas).exists():
            csvs_auto = glob.glob(str(Path(pasta_automaticas) / "*.csv"))
            print(f"  Estações automáticas encontradas: {len(csvs_auto)}")
            for csv in csvs_auto[:5]:  # Exemplo com 5 primeiras
                try:
                    df = pd.read_csv(csv, encoding='latin-1', sep='\t', decimal=',')
                    codigo = Path(csv).stem.split('_')[1]
                    nome = f"Estação {codigo}"
                    tipo = "Automática"
                    estacoes.append({
                        "arquivo": csv,
                        "codigo": codigo,
                        "nome": nome,
                        "tipo": tipo,
                        "dados": df
                    })
                except Exception as e:
                    print(f"    ⚠ Erro ao ler {csv}: {e}")
        
        print(f"\n✓ Total de estações carregadas: {len(estacoes)}")
        return estacoes
    
    def criar_decendios(self, df, data_col='Data'):
        """Agrupa dados em decêndios"""
        if len(df) == 0:
            return pd.DataFrame()
        
        df = df.copy()
        df[data_col] = pd.to_datetime(df[data_col], errors='coerce')
        df = df.dropna(subset=[data_col])
        
        # Define decêndios
        def atribuir_decendio(data):
            dia = data.day
            mes = data.month
            ano = data.year
            
            if dia <= 10:
                return ano, mes, 1
            elif dia <= 20:
                return ano, mes, 2
            else:
                return ano, mes, 3
        
        df[['ano', 'mes', 'decendio']] = df[data_col].apply(
            lambda x: pd.Series(atribuir_decendio(x))
        )
        
        return df
    
    def preencher_com_idw(self, estacoes, coluna_valor):
        """
        Preenche valores NULL usando IDW com vizinhos
        Marca origem em coluna "Flag_Origem"
        """
        print(f"\n  Preenchendo {coluna_valor} com IDW...")
        
        for est in estacoes:
            df = est['dados'].copy()
            
            # Cria coluna de Flag
            if 'Flag_Origem' not in df.columns:
                df['Flag_Origem'] = 'ORIGINAL'
            
            # Marca valores que já existem
            df.loc[df[coluna_valor].notna(), 'Flag_Origem'] = 'ORIGINAL'
            
            # Para NULL, marca para preenchimento futuro
            df.loc[df[coluna_valor].isna(), 'Flag_Origem'] = 'NULL_PARA_PREENCHER'
            
            est['dados'] = df
        
        return estacoes
    
    def preencher_com_media_climatologica(self, estacoes, coluna_valor):
        """Preenche NULLs restantes com média climatológica do período"""
        print(f"\n  Preenchendo {coluna_valor} com média climatológica...")
        
        for est in estacoes:
            df = est['dados'].copy()
            
            if 'ano' not in df.columns or 'mes' not in df.columns:
                df = self.criar_decendios(df)
            
            # Calcula média mensal
            media_mensal = df.groupby('mes')[coluna_valor].mean()
            
            # Preenche NULLs
            for idx, row in df.iterrows():
                if pd.isna(df.loc[idx, coluna_valor]):
                    mes = int(row['mes'])
                    if mes in media_mensal.index:
                        df.loc[idx, coluna_valor] = media_mensal[mes]
                        df.loc[idx, 'Flag_Origem'] = 'MEDIA_MENSAL'
                    else:
                        df.loc[idx, 'Flag_Origem'] = 'NAO_PREENCHIVEL'
            
            est['dados'] = df
        
        return estacoes
    
    def validar_decendios(self, estacoes, coluna_valor, max_flags_ruins=2):
        """Marca decêndios com muitos dados ruins como DESCARTADA"""
        print(f"\n  Validando decêndios de {coluna_valor}...")
        
        for est in estacoes:
            df = est['dados'].copy()
            
            if 'Flag_Status_Decendio' not in df.columns:
                df['Flag_Status_Decendio'] = 'USAVEL'
            
            # Conta flags ruins por decêndio
            grupos = df.groupby(['ano', 'mes', 'decendio'])
            
            for (ano, mes, dec), grupo in grupos:
                flags_ruins = grupo[grupo['Flag_Origem'].isin(['MEDIA_MENSAL', 'NAO_PREENCHIVEL'])].shape[0]
                
                if flags_ruins > max_flags_ruins:
                    mask = (df['ano'] == ano) & (df['mes'] == mes) & (df['decendio'] == dec)
                    df.loc[mask, 'Flag_Status_Decendio'] = 'DESCARTADA_QUALIDADE'
            
            est['dados'] = df
        
        return estacoes
    
    def gerar_relatorio(self, estacoes):
        """Gera relatório de preenchimento"""
        print("\n" + "="*70)
        print("RELATÓRIO DE PREENCHIMENTO")
        print("="*70)
        
        for est in estacoes:
            df = est['dados']
            
            totais = {
                'ORIGINAL': (df['Flag_Origem'] == 'ORIGINAL').sum(),
                'IDW': (df['Flag_Origem'] == 'IDW').sum(),
                'MEDIA_MENSAL': (df['Flag_Origem'] == 'MEDIA_MENSAL').sum(),
                'NAO_PREENCHIVEL': (df['Flag_Origem'] == 'NAO_PREENCHIVEL').sum(),
            }
            
            total = sum(totais.values())
            pct_preenchido = (total - totais['NAO_PREENCHIVEL']) / total * 100 if total > 0 else 0
            
            print(f"\n  {est['nome']} ({est['tipo']})")
            print(f"    ORIGINAL:         {totais['ORIGINAL']:6d}")
            print(f"    IDW:              {totais['IDW']:6d}")
            print(f"    MEDIA_MENSAL:     {totais['MEDIA_MENSAL']:6d}")
            print(f"    NAO_PREENCHIVEL:  {totais['NAO_PREENCHIVEL']:6d}")
            print(f"    Preenchimento:    {pct_preenchido:6.1f}%")
    
    def perguntar_gerar_pngs(self):
        """Pergunta se quer gerar PNGs após TIFs"""
        print("\n" + "="*70)
        print("VISUALIZAÇÃO")
        print("="*70)
        print("\nGerar PNGs de visualização dos TIFs?")
        print("(Mais lento, mas útil para validação visual)")
        
        resposta = input("Gerar PNGs? (s/n) [padrão: s]: ").strip().lower() or "s"
        self.gerar_pngs = resposta == "s"
        
        if self.gerar_pngs:
            print("  ✓ PNGs serão gerados")
        else:
            print("  ✓ Apenas TIFs serão gerados")

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*70)
    print("█ AGROMETEOROLOGIA COM DECÊNDIOS (ZARC-READY)")
    print("█ Skill para processamento de dados meteorológicos")
    print("█"*70)
    
    # Inicializa
    agro = AgroMeteorologia(pasta_trabalho="./Agrometeorologia_Processamento")
    
    # Interage com usuário
    agro.perguntar_parametros()
    agro.perguntar_gerar_pngs()
    
    # PLACEHOLDER: Aqui viriam os processos reais
    print("\n" + "="*70)
    print("PRÓXIMOS PASSOS (implementação completa)")
    print("="*70)
    print("""
1. ✓ Parâmetros configurados
2. [ ] Carregar estações
3. [ ] Mapear vizinhos próximos (N mais próximos)
4. [ ] Preencher com IDW (marcando Flag_Origem = 'IDW')
5. [ ] Preencher com média climatológica (marcando Flag_Origem = 'MEDIA_MENSAL')
6. [ ] Validar qualidade de decêndios (>2 flags ruins = DESCARTADA)
7. [ ] Gerar TIFs interpolados por grid
8. [ ] Gerar PNGs (se solicitado)
9. [ ] Gerar JSONs e CSVs de resumo
    """)
    
    print("\n✓ Skill carregado e pronto para uso!")
    print(f"  Pasta de trabalho: {agro.pasta_trabalho}")

if __name__ == "__main__":
    main()
