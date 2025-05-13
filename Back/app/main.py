from flask import Flask, request, jsonify
from google.cloud import storage
import os, tempfile
import time  

app = Flask(__name__)

# Asegúrate de tener GOOGLE_APPLICATION_CREDENTIALS apuntando a tu JSON de servicio
BUCKET_NAME = "pantherkit"

# Definición de personalidades STEM
STEM_PERSONALITIES = {
    "S": "Science (Ciencia)",
    "T": "Technology (Tecnología)",
    "E": "Engineering (Ingeniería)",
    "M": "Mathematics (Matemáticas)"
}

# Mapeo de carreras a cada categoría STEM
degree_map = {
    'ciencia': ['Mecánica eléctrica'],
    'tecnologia': ['Desarrollo de software', 'Computación'],
    'ingenieria': ['Mecatrónica', 'Robótica'],
    'matematicas': ['Electrónica y Computación']
}

@app.route('/predict_stem_personality', methods=['POST'])
def predict_stem_personality():
    """
    Recibe un JSON con las respuestas del usuario y predice su personalidad STEM y su carrera sugerida.
    """
    try:
        # Obtener datos del request
        data = request.get_json()

        if not data or 'respuestas' not in data:
            return jsonify({"error": "Formato de datos inválido. Se requiere un objeto JSON con el campo 'respuestas'."}), 400

        respuestas = data['respuestas']

        # Verificar que todas las categorías STEM estén presentes
        categorias_requeridas = ['ciencia', 'tecnologia', 'ingenieria', 'matematicas']
        for categoria in categorias_requeridas:
            if categoria not in respuestas:
                return jsonify({"error": f"Falta la categoría '{categoria}' en las respuestas."}), 400

        # Calcular puntuación promedio para cada categoría
        promedios = {}
        for categoria in categorias_requeridas:
            if not respuestas[categoria] or not isinstance(respuestas[categoria], list):
                return jsonify({"error": f"La categoría '{categoria}' debe ser una lista de puntuaciones."}), 400

            # Verificar que todas las puntuaciones sean números entre 1 y 5
            for puntuacion in respuestas[categoria]:
                if not isinstance(puntuacion, (int, float)) or puntuacion < 1 or puntuacion > 5:
                    return jsonify({"error": f"Todas las puntuaciones deben ser números entre 1 y 5."}), 400

            promedios[categoria] = sum(respuestas[categoria]) / len(respuestas[categoria])

        # Determinar la personalidad STEM basada en la categoría con mayor puntuación
        categoria_max = max(promedios, key=promedios.get)
        
        # Verificar si hay empate entre las categorías con la mayor puntuación
        max_score = max(promedios.values())
        top_categories = [cat for cat, score in promedios.items() if score == max_score]

        # Si hay un empate entre categorías, sugerir "No se ha encontrado una carrera"
        if len(top_categories) > 1:
            return jsonify({
                "personalidad_stem": "No clasificado",
                "descripcion": "No se ha encontrado una carrera adecuada en este momento, pero pronto tendremos más opciones.",
                "puntuaciones": promedios,
                "detalles": "Tu perfil STEM no se clasifica en una sola categoría definida.",
                "carreras": ["Aún no tenemos una carrera definida para ti."]
            }), 200

        # Mapear la categoría a la personalidad STEM
        stem_map = {
            'ciencia': 'S',
            'tecnologia': 'T',
            'ingenieria': 'E',
            'matematicas': 'M'
        }

        personalidad_stem = stem_map[categoria_max]
        descripcion_stem = STEM_PERSONALITIES[personalidad_stem]

        # Sugerir carreras basadas en la categoría con mayor puntuación
        suggested_degrees = degree_map.get(categoria_max, [])

        # Preparar resultados detallados
        resultados = {
            "personalidad_stem": personalidad_stem,
            "descripcion": descripcion_stem,
            "puntuaciones": promedios,
            "detalles": {
                "S": {
                    "nombre": STEM_PERSONALITIES["S"],
                    "puntuacion": promedios['ciencia'],
                    "porcentaje": round(promedios['ciencia'] / 5 * 100, 2)
                },
                "T": {
                    "nombre": STEM_PERSONALITIES["T"],
                    "puntuacion": promedios['tecnologia'],
                    "porcentaje": round(promedios['tecnologia'] / 5 * 100, 2)
                },
                "E": {
                    "nombre": STEM_PERSONALITIES["E"],
                    "puntuacion": promedios['ingenieria'],
                    "porcentaje": round(promedios['ingenieria'] / 5 * 100, 2)
                },
                "M": {
                    "nombre": STEM_PERSONALITIES["M"],
                    "puntuacion": promedios['matematicas'],
                    "porcentaje": round(promedios['matematicas'] / 5 * 100, 2)
                }
            },
            "carreras": suggested_degrees
        }

        return jsonify(resultados), 200

    except Exception as e:
        return jsonify({"error": f"Error al procesar la solicitud: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Permite conexiones externas
