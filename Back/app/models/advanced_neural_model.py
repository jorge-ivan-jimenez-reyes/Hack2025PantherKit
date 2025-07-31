import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Input, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Embedding, Concatenate, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("advanced_neural_model")

class TransformerBlock(tf.keras.layers.Layer):
    """
    Bloque Transformer personalizado con Multi-Head Attention y Feed Forward Network.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

class AdvancedNeuralCareerModel:
    """
    Modelo neural de última generación usando arquitectura Transformer
    para recomendación de carreras basada en perfiles MBTI y MI.
    """
    
    def __init__(self):
        """Inicializa el modelo neural avanzado."""
        self.model = None
        self.label_encoder = LabelEncoder()
        self.model_path = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "data" / "advanced_neural_models"
        os.makedirs(self.model_path, exist_ok=True)
        
        # Parámetros del modelo
        self.embed_dim = 128  # Dimensión de embedding
        self.num_heads = 8    # Número de cabezas de atención
        self.ff_dim = 256     # Dimensión de feed-forward
        self.num_transformer_blocks = 4  # Número de bloques transformer
        
        # Intentar cargar modelo existente
        self.load_model()
        
    def create_advanced_architecture(self, input_dim: int, num_classes: int) -> Model:
        """
        Crea una arquitectura neural avanzada usando Transformers.
        
        Args:
            input_dim: Dimensión de entrada (16 para MBTI+MI)
            num_classes: Número de clases de salida (carreras)
            
        Returns:
            Modelo compilado
        """
        # Input layer
        inputs = Input(shape=(input_dim,), name="feature_input")
        
        # Embedding inicial para convertir features a representación densa
        x = Dense(self.embed_dim, activation="gelu", name="initial_embedding")(inputs)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(0.1)(x)
        
        # Reshape para trabajar con secuencias (tratamos cada feature como un token)
        x = tf.expand_dims(x, axis=1)  # (batch_size, 1, embed_dim)
        
        # Positional encoding (simple suma de posición)
        pos_encoding = self.get_positional_encoding(1, self.embed_dim)
        x = x + pos_encoding
        
        # Stack de bloques Transformer
        for i in range(self.num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                rate=0.1,
                name=f"transformer_block_{i}"
            )(x)
        
        # Global pooling para reducir secuencia a vector
        x = GlobalAveragePooling1D()(x)
        
        # Procesamiento adicional con capas densas
        x = Dense(256, activation="gelu", name="dense_1")(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(0.2)(x)
        
        x = Dense(128, activation="gelu", name="dense_2")(x)
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dropout(0.1)(x)
        
        # Capa de salida
        outputs = Dense(num_classes, activation="softmax", name="career_output")(x)
        
        # Crear modelo
        model = Model(inputs=inputs, outputs=outputs, name="AdvancedCareerPredictor")
        
        # Compilar con optimizador Adam (AdamW no disponible en TF 2.11)
        optimizer = Adam(
            learning_rate=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_3_categorical_accuracy']
        )
        
        return model
    
    def get_positional_encoding(self, seq_len: int, d_model: int) -> tf.Tensor:
        """
        Genera encoding posicional para el Transformer.
        """
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        
        # Aplicar sin a índices pares y cos a índices impares
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def prepare_input_features(self, mbti_vector: List[int], mbti_weights: Dict[str, float], 
                              mi_scores: Dict[str, float]) -> np.ndarray:
        """
        Prepara las características de entrada para el modelo.
        
        Args:
            mbti_vector: Vector binario de MBTI [0,1,0,1]
            mbti_weights: Pesos de dimensiones MBTI
            mi_scores: Puntuaciones de inteligencias múltiples
            
        Returns:
            Vector numpy normalizado de características
        """
        # Convertir mbti_weights a vector
        mbti_dimensions = ["E/I", "S/N", "T/F", "J/P"]
        mbti_weight_vector = [mbti_weights[dim] for dim in mbti_dimensions]
        
        # Convertir mi_scores a vector
        mi_types = ["Lin", "LogMath", "Spa", "BodKin", "Mus", "Inter", "Intra", "Nat"]
        mi_vector = [mi_scores.get(mi_type, 0.0) for mi_type in mi_types]
        
        # Combinar todos los vectores
        combined_vector = np.array(mbti_vector + mbti_weight_vector + mi_vector, dtype=np.float32)
        
        # Normalización Z-score para mejor rendimiento del Transformer
        combined_vector = (combined_vector - np.mean(combined_vector)) / (np.std(combined_vector) + 1e-8)
        
        return combined_vector.reshape(1, -1)
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   validation_split: float = 0.2, epochs: int = 100, 
                   batch_size: int = 64) -> Dict:
        """
        Entrena el modelo neural avanzado.
        
        Args:
            X: Matriz de características (N x 16)
            y: Etiquetas de carreras (índices enteros)
            validation_split: Proporción para validación
            epochs: Número máximo de epochs
            batch_size: Tamaño del batch
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        logger.info("Iniciando entrenamiento del modelo neural avanzado...")
        
        # Preparar etiquetas
        if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y
            
        num_classes = len(np.unique(y_encoded))
        input_dim = X.shape[1]
        
        logger.info(f"Dimensiones: {X.shape}, Clases: {num_classes}")
        
        # Crear modelo si no existe
        if self.model is None:
            self.model = self.create_advanced_architecture(input_dim, num_classes)
            
        # Mostrar resumen del modelo
        logger.info("Arquitectura del modelo:")
        self.model.summary(print_fn=logger.info)
        
        # Callbacks avanzados
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.model_path / "best_model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        # Normalizar datos de entrada
        X_normalized = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        
        # Entrenar modelo
        logger.info("Iniciando entrenamiento...")
        history = self.model.fit(
            X_normalized, y_encoded,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Guardar modelo y encoder
        self.save_model()
        
        # Calcular métricas finales
        val_loss = min(history.history['val_loss'])
        val_accuracy = max(history.history['val_accuracy'])
        
        logger.info(f"Entrenamiento completado. Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return {
            "message": "Modelo entrenado exitosamente",
            "final_val_loss": float(val_loss),
            "final_val_accuracy": float(val_accuracy),
            "epochs_trained": len(history.history['loss']),
            "num_classes": num_classes
        }
    
    def predict_career(self, mbti_vector: List[int], mbti_weights: Dict[str, float], 
                      mi_scores: Dict[str, float], career_names: List[str]) -> List[Tuple[str, float]]:
        """
        Predice carreras usando el modelo entrenado.
        
        Args:
            mbti_vector: Vector MBTI del usuario
            mbti_weights: Pesos MBTI
            mi_scores: Puntuaciones MI
            career_names: Lista de nombres de carreras
            
        Returns:
            Lista de tuplas (carrera, probabilidad) ordenadas por probabilidad
        """
        if self.model is None:
            raise ValueError("El modelo no está entrenado. Entrena el modelo primero.")
            
        # Preparar entrada
        X = self.prepare_input_features(mbti_vector, mbti_weights, mi_scores)
        
        # Hacer predicción
        predictions = self.model.predict(X, verbose=0)[0]
        
        # Combinar con nombres de carreras
        results = [(career, float(prob)) for career, prob in zip(career_names, predictions)]
        
        # Ordenar por probabilidad descendente
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evalúa el modelo en datos de prueba.
        """
        if self.model is None:
            raise ValueError("El modelo no está entrenado.")
            
        # Normalizar datos de prueba
        X_test_normalized = (X_test - np.mean(X_test, axis=0)) / (np.std(X_test, axis=0) + 1e-8)
        
        # Evaluar
        results = self.model.evaluate(X_test_normalized, y_test, verbose=0)
        
        return {
            "test_loss": float(results[0]),
            "test_accuracy": float(results[1]),
            "test_top3_accuracy": float(results[2]) if len(results) > 2 else None
        }
    
    def save_model(self):
        """Guarda el modelo y el encoder de etiquetas."""
        if self.model:
            self.model.save(str(self.model_path / "advanced_model.h5"))
            logger.info(f"Modelo guardado en {self.model_path / 'advanced_model.h5'}")
            
        if hasattr(self.label_encoder, 'classes_') and self.label_encoder.classes_.size > 0:
            joblib.dump(self.label_encoder, str(self.model_path / "label_encoder.pkl"))
            logger.info("Encoder de etiquetas guardado")
    
    def load_model(self):
        """Carga el modelo entrenado si existe."""
        try:
            model_path = self.model_path / "advanced_model.h5"
            if model_path.exists():
                # Registrar clases personalizadas
                custom_objects = {
                    'TransformerBlock': TransformerBlock,
                    'gelu': tf.keras.activations.gelu
                }
                self.model = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
                logger.info("Modelo neural avanzado cargado exitosamente")
                
            encoder_path = self.model_path / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = joblib.load(str(encoder_path))
                logger.info("Encoder de etiquetas cargado")
                
        except Exception as e:
            logger.warning(f"Error al cargar el modelo: {e}")
            self.model = None
    
    def get_model_summary(self) -> str:
        """Retorna un resumen del modelo como string."""
        if self.model is None:
            return "Modelo no inicializado"
            
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()

if __name__ == "__main__":
    # Ejemplo de uso
    model = AdvancedNeuralCareerModel()
    print("Modelo neural avanzado inicializado")
    
    # Generar datos de ejemplo
    X_example = np.random.rand(1000, 16)
    y_example = np.random.randint(0, 10, 1000)
    
    # Entrenar modelo
    results = model.train_model(X_example, y_example, epochs=5)
    print(f"Resultados: {results}") 