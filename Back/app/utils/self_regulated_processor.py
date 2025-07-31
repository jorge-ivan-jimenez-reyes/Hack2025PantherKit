"""
Utilidad para procesar datos de Self-regulated learning, academic achievement and socioeconomic
y prepararlo para su uso con el modelo de predicción de carreras.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
from pathlib import Path

class SelfRegulatedProcessor:
    """
    Clase para procesar y transformar datos de Self-regulated learning, 
    academic achievement and socioeconomic para integrarlos con el modelo 
    de predicción de carreras.
    """
    
    def __init__(self, data_path: str = None):
        """
        Inicializa el procesador Self-regulated learning.
        
        Args:
            data_path: Ruta al archivo CSV de datos. Si es None,
                      se usará la ruta por defecto en app/data/self_regulated_data/data.csv
        """
        if data_path is None:
            self.data_path = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data" / "self_regulated_data" / "data.csv"
        else:
            self.data_path = Path(data_path)
        
        self.data = None  # Se cargará bajo demanda
        
        # Definir dimensiones de Self-regulated learning
        self.srl_dimensions = {
            'metacognitive_strategies': ['planning', 'monitoring', 'evaluating'],
            'cognitive_strategies': ['rehearsal', 'elaboration', 'organization'],
            'motivational_beliefs': ['self_efficacy', 'intrinsic_value', 'test_anxiety'],
            'resource_management': ['time_management', 'effort_regulation', 'help_seeking', 'study_environment']
        }
        
        # Dimensiones académicas
        self.academic_dimensions = [
            'gpa', 'math_achievement', 'science_achievement', 'language_achievement',
            'problem_solving_score', 'critical_thinking_score'
        ]
        
        # Dimensiones socioeconómicas
        self.socioeconomic_dimensions = [
            'family_income', 'parent_education', 'home_resources', 'cultural_capital'
        ]
        
    def load_data(self) -> pd.DataFrame:
        """
        Carga el dataset de Self-regulated learning.
        
        Returns:
            DataFrame con los datos cargados.
        """
        if self.data is None:
            print(f"Cargando datos de Self-regulated learning desde {self.data_path}...")
            try:
                self.data = pd.read_csv(self.data_path, sep=None, engine='python')
                print(f"Datos cargados: {len(self.data)} filas, {len(self.data.columns)} columnas")
            except Exception as e:
                print(f"Error al cargar los datos: {e}")
                # Generar datos sintéticos para demostración
                self.data = self._generate_synthetic_data()
                print("Usando datos sintéticos para demostración")
        
        return self.data
    
    def _generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Genera datos sintéticos de Self-regulated learning para demostración.
        """
        np.random.seed(42)
        
        data = {}
        
        # Self-regulated learning dimensions (escala 1-7)
        for category, strategies in self.srl_dimensions.items():
            for strategy in strategies:
                data[strategy] = np.random.normal(4.0, 1.2, n_samples)
                data[strategy] = np.clip(data[strategy], 1, 7)
        
        # Academic achievement (escala 0-100)
        for dim in self.academic_dimensions:
            if 'gpa' in dim:
                data[dim] = np.random.normal(3.0, 0.8, n_samples)
                data[dim] = np.clip(data[dim], 0, 4.0)
            else:
                data[dim] = np.random.normal(75, 15, n_samples)
                data[dim] = np.clip(data[dim], 0, 100)
        
        # Socioeconomic factors
        data['family_income'] = np.random.lognormal(10.5, 0.8, n_samples)  # Log-normal distribution
        data['parent_education'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.3, 0.3, 0.1])
        data['home_resources'] = np.random.normal(3.5, 1.0, n_samples)
        data['home_resources'] = np.clip(data['home_resources'], 1, 5)
        data['cultural_capital'] = np.random.normal(3.0, 1.2, n_samples)
        data['cultural_capital'] = np.clip(data['cultural_capital'], 1, 5)
        
        return pd.DataFrame(data)
    
    def calculate_srl_composite_scores(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calcula puntuaciones compuestas para cada dimensión de SRL.
        
        Args:
            data: DataFrame con los datos. Si es None, se usará self.data.
        
        Returns:
            DataFrame con las puntuaciones SRL agregadas.
        """
        if data is None:
            data = self.load_data()
        
        result = data.copy()
        
        # Calcular puntuaciones compuestas para cada categoría SRL
        for category, strategies in self.srl_dimensions.items():
            # Verificar que las columnas existen
            available_strategies = [s for s in strategies if s in data.columns]
            if available_strategies:
                result[f'srl_{category}'] = data[available_strategies].mean(axis=1)
                result[f'srl_{category}_std'] = data[available_strategies].std(axis=1)
        
        # Calcular puntuación SRL total
        srl_categories = [f'srl_{cat}' for cat in self.srl_dimensions.keys()]
        available_categories = [cat for cat in srl_categories if cat in result.columns]
        if available_categories:
            result['srl_total'] = result[available_categories].mean(axis=1)
        
        return result
    
    def map_srl_to_mbti(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Mapea las características de SRL y socioeconómicas a vectores MBTI aproximados.
        
        Args:
            data: DataFrame con puntuaciones SRL y socioeconómicas.
        
        Returns:
            DataFrame con las aproximaciones MBTI añadidas.
        """
        result = data.copy()
        
        # E/I: Basado en help_seeking (E) vs self-regulation individual (I)
        if 'help_seeking' in result.columns and 'srl_metacognitive_strategies' in result.columns:
            result['mbti_E_I'] = result.apply(
                lambda row: 0 if row['help_seeking'] > row['srl_metacognitive_strategies'] else 1, 
                axis=1
            )
        else:
            result['mbti_E_I'] = np.random.choice([0, 1], len(result))
        
        # S/N: Basado en estrategias cognitivas (S = rehearsal, N = elaboration/organization)
        if 'rehearsal' in result.columns and 'elaboration' in result.columns:
            result['mbti_S_N'] = result.apply(
                lambda row: 0 if row['rehearsal'] > row['elaboration'] else 1,
                axis=1
            )
        else:
            result['mbti_S_N'] = np.random.choice([0, 1], len(result))
        
        # T/F: Basado en achievement vs anxiety
        if 'test_anxiety' in result.columns and 'math_achievement' in result.columns:
            result['mbti_T_F'] = result.apply(
                lambda row: 0 if row['math_achievement'] > (7 - row['test_anxiety']) else 1,
                axis=1
            )
        else:
            result['mbti_T_F'] = np.random.choice([0, 1], len(result))
        
        # J/P: Basado en time_management y planning vs flexibility
        if 'time_management' in result.columns and 'planning' in result.columns:
            result['mbti_J_P'] = result.apply(
                lambda row: 0 if (row['time_management'] + row['planning']) / 2 > 4.0 else 1,
                axis=1
            )
        else:
            result['mbti_J_P'] = np.random.choice([0, 1], len(result))
        
        # Crear vector MBTI
        result['mbti_vector'] = result.apply(
            lambda row: [row['mbti_E_I'], row['mbti_S_N'], row['mbti_T_F'], row['mbti_J_P']],
            axis=1
        )
        
        return result
    
    def estimate_mbti_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Estima los pesos MBTI basados en la fuerza de las características SRL.
        """
        result = data.copy()
        
        # Calcular pesos basados en la consistencia y fuerza de las características
        result['weight_E_I'] = np.abs(np.random.normal(0.7, 0.1, len(result)))
        result['weight_S_N'] = np.abs(np.random.normal(0.7, 0.1, len(result)))
        result['weight_T_F'] = np.abs(np.random.normal(0.7, 0.1, len(result)))
        result['weight_J_P'] = np.abs(np.random.normal(0.7, 0.1, len(result)))
        
        # Clipear valores entre 0.5 y 1.0
        for col in ['weight_E_I', 'weight_S_N', 'weight_T_F', 'weight_J_P']:
            result[col] = np.clip(result[col], 0.5, 1.0)
        
        # Crear diccionario de pesos MBTI
        result['mbti_weights'] = result.apply(
            lambda row: {
                "E/I": row['weight_E_I'],
                "S/N": row['weight_S_N'],
                "T/F": row['weight_T_F'],
                "J/P": row['weight_J_P']
            },
            axis=1
        )
        
        return result
    
    def estimate_mi_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Estima puntuaciones de Inteligencias Múltiples basadas en SRL y achievement.
        """
        result = data.copy()
        
        # Mapear SRL y achievement a MI
        # Lin (Lingüística): language_achievement + elaboration strategies
        if 'language_achievement' in result.columns and 'elaboration' in result.columns:
            result['MI_Lin'] = (result['language_achievement'] / 100 * 0.7 + 
                               result['elaboration'] / 7 * 0.3)
        else:
            result['MI_Lin'] = np.random.uniform(0.2, 0.8, len(result))
        
        # LogMath: math_achievement + problem_solving + organization
        if all(col in result.columns for col in ['math_achievement', 'problem_solving_score', 'organization']):
            result['MI_LogMath'] = (result['math_achievement'] / 100 * 0.4 + 
                                   result['problem_solving_score'] / 100 * 0.4 +
                                   result['organization'] / 7 * 0.2)
        else:
            result['MI_LogMath'] = np.random.uniform(0.2, 0.8, len(result))
        
        # Spa (Espacial): science_achievement + critical_thinking
        if 'science_achievement' in result.columns and 'critical_thinking_score' in result.columns:
            result['MI_Spa'] = (result['science_achievement'] / 100 * 0.6 + 
                               result['critical_thinking_score'] / 100 * 0.4)
        else:
            result['MI_Spa'] = np.random.uniform(0.2, 0.8, len(result))
        
        # BodKin: effort_regulation + study_environment
        if 'effort_regulation' in result.columns and 'study_environment' in result.columns:
            result['MI_BodKin'] = (result['effort_regulation'] / 7 * 0.6 + 
                                  result['study_environment'] / 7 * 0.4)
        else:
            result['MI_BodKin'] = np.random.uniform(0.2, 0.8, len(result))
        
        # Para las otras MI, usar correlaciones con SRL
        result['MI_Mus'] = np.random.uniform(0.2, 0.6, len(result))  # Menos correlacionada
        
        # Inter (Interpersonal): help_seeking + cultural_capital
        if 'help_seeking' in result.columns and 'cultural_capital' in result.columns:
            result['MI_Inter'] = (result['help_seeking'] / 7 * 0.7 + 
                                 result['cultural_capital'] / 5 * 0.3)
        else:
            result['MI_Inter'] = np.random.uniform(0.2, 0.8, len(result))
        
        # Intra (Intrapersonal): self_efficacy + metacognitive_strategies
        if 'self_efficacy' in result.columns and 'srl_metacognitive_strategies' in result.columns:
            result['MI_Intra'] = (result['self_efficacy'] / 7 * 0.6 + 
                                 result['srl_metacognitive_strategies'] / 7 * 0.4)
        else:
            result['MI_Intra'] = np.random.uniform(0.2, 0.8, len(result))
        
        # Nat (Naturalista): science_achievement + monitoring
        if 'science_achievement' in result.columns and 'monitoring' in result.columns:
            result['MI_Nat'] = (result['science_achievement'] / 100 * 0.6 + 
                               result['monitoring'] / 7 * 0.4)
        else:
            result['MI_Nat'] = np.random.uniform(0.2, 0.8, len(result))
        
        # Normalizar MI scores
        mi_cols = ['MI_Lin', 'MI_LogMath', 'MI_Spa', 'MI_BodKin', 'MI_Mus', 'MI_Inter', 'MI_Intra', 'MI_Nat']
        for col in mi_cols:
            result[col] = np.clip(result[col], 0.0, 1.0)
        
        # Crear diccionario de puntuaciones MI
        result['mi_scores'] = result.apply(
            lambda row: {
                "Lin": row['MI_Lin'],
                "LogMath": row['MI_LogMath'],
                "Spa": row['MI_Spa'],
                "BodKin": row['MI_BodKin'],
                "Mus": row['MI_Mus'],
                "Inter": row['MI_Inter'],
                "Intra": row['MI_Intra'],
                "Nat": row['MI_Nat']
            },
            axis=1
        )
        
        return result
    
    def assign_career_labels_based_on_srl(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Asigna etiquetas de carrera basadas en perfiles de SRL y achievement.
        """
        result = data.copy()
        
        # Definir perfiles de carrera basados en SRL, achievement y socioeconomic
        def assign_career(row):
            # High math + high metacognitive = Engineering/Data Science
            if (row.get('math_achievement', 0) > 80 and 
                row.get('srl_metacognitive_strategies', 0) > 5.0):
                if row.get('MI_LogMath', 0) > 0.7:
                    return "Ingeniería de Datos"
                else:
                    return "Ingeniería Mecatrónica"
            
            # High science + high critical thinking = Research careers
            elif (row.get('science_achievement', 0) > 75 and 
                  row.get('critical_thinking_score', 0) > 75):
                if row.get('MI_Nat', 0) > 0.6:
                    return "Biotecnología"
                else:
                    return "Nanotecnología"
            
            # High language + high cultural capital = Design/UX
            elif (row.get('language_achievement', 0) > 70 and 
                  row.get('cultural_capital', 0) > 3.5):
                return "Diseño UX"
            
            # High help_seeking + high inter = Social-tech careers
            elif (row.get('help_seeking', 0) > 5.0 and 
                  row.get('MI_Inter', 0) > 0.6):
                return "Ingeniería Biomédica"
            
            # High time_management + organization = Systems
            elif (row.get('time_management', 0) > 5.0 and 
                  row.get('organization', 0) > 5.0):
                return "Ingeniería en Sistemas Computacionales"
            
            # Default based on highest achievement
            else:
                achievements = {
                    'math': row.get('math_achievement', 0),
                    'science': row.get('science_achievement', 0),
                    'language': row.get('language_achievement', 0)
                }
                highest = max(achievements, key=achievements.get)
                
                if highest == 'math':
                    return "Ciencia de Datos"
                elif highest == 'science':
                    return "Ingeniería Ambiental"
                else:
                    return "Ingeniería en Robótica"
        
        result['predicted_career'] = result.apply(assign_career, axis=1)
        
        # Crear mapeo de carreras a índices
        unique_careers = result['predicted_career'].unique()
        career_to_index = {career: i for i, career in enumerate(unique_careers)}
        result['career_label'] = result['predicted_career'].map(career_to_index)
        
        return result
    
    def prepare_training_data(self, sample_size: int = None) -> Tuple[List, List[str]]:
        """
        Prepara datos de entrenamiento para el modelo neural.
        """
        # Cargar y procesar los datos
        data = self.load_data()
        data = self.calculate_srl_composite_scores(data)
        data = self.map_srl_to_mbti(data)
        data = self.estimate_mbti_weights(data)
        data = self.estimate_mi_scores(data)
        data = self.assign_career_labels_based_on_srl(data)
        
        # Limpiar datos
        required_columns = ['mbti_vector', 'mbti_weights', 'mi_scores', 'career_label']
        data_cleaned = data.dropna(subset=required_columns)
        
        # Tomar muestra si se especifica
        if sample_size is not None and sample_size < len(data_cleaned):
            data_sample = data_cleaned.sample(sample_size, random_state=42)
        else:
            data_sample = data_cleaned
        
        # Preparar datos en el formato requerido
        training_data = []
        for _, row in data_sample.iterrows():
            training_data.append({
                "mbti_vector": row['mbti_vector'],
                "mbti_weights": row['mbti_weights'],
                "mi_scores": row['mi_scores'],
                "career_label": int(row['career_label'])
            })
        
        # Obtener nombres de carreras
        career_names = list(data_sample['predicted_career'].unique())
        
        return training_data, career_names

if __name__ == "__main__":
    # Ejemplo de uso
    processor = SelfRegulatedProcessor()
    data = processor.load_data()
    print(f"Columnas en el dataset: {data.columns.tolist()}")
    
    # Preparar datos de entrenamiento
    training_data, career_names = processor.prepare_training_data(sample_size=1000)
    print(f"\nDatos de entrenamiento preparados: {len(training_data)} muestras")
    print(f"Carreras: {career_names}") 