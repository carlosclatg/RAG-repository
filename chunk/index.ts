/**
 * Divide un texto en fragmentos (chunks) respetando párrafos y frases.
 */
export function chunkText(
  text: string, 
  maxChunkSize: number = 1000, 
  minChunkSize: number = 100
): string[] {
  // 1. Dividir por saltos de línea (párrafos) y limpiar espacios vacíos
  const paragraphs: string[] = text.split(/\n\s*\n/)
    .map(p => p.replace(/\s+/g, " ").trim())
    .filter(p => p.length > 0);

  const chunks: string[] = [];
  let currentChunk: string = "";

  for (const paragraph of paragraphs) {
    // Si un solo párrafo es excesivamente largo, lo dividimos por frases
    if (paragraph.length > maxChunkSize) {
      if (currentChunk) chunks.push(currentChunk);
      
      // El operador || [] es vital aquí por si match devuelve null
      const sentences: string[] = paragraph.match(/[^.!?]+[.!?]+/g) || [paragraph];
      let subChunk: string = "";
      
      for (const sentence of sentences) {
        if ((subChunk + sentence).length > maxChunkSize) {
          if (subChunk) chunks.push(subChunk.trim());
          subChunk = sentence;
        } else {
          subChunk += (subChunk ? " " : "") + sentence;
        }
      }
      currentChunk = subChunk;
      continue;
    }

    // Si añadir el párrafo actual supera el máximo, guardamos el chunk actual
    if ((currentChunk + paragraph).length > maxChunkSize) {
      if (currentChunk.length >= minChunkSize) {
        chunks.push(currentChunk.trim());
        currentChunk = paragraph;
      } else {
        // Si el chunk actual es muy pequeño, forzamos la unión
        currentChunk += (currentChunk ? " " : "") + paragraph;
      }
    } else {
      currentChunk += (currentChunk ? "\n\n" : "") + paragraph;
    }
  }

  if (currentChunk) chunks.push(currentChunk.trim());
  return chunks;
}

/**
 * Divide el texto de forma recursiva (sliding window) con solapamiento (overlap).
 */
export function recursiveChunkingBySentences(
  text: string, 
  maxSize: number = 800, 
  overlap: number = 150
): string[] {
  const chunks: string[] = [];
  let currentIndex: number = 0;

  while (currentIndex < text.length) {
    let endIndex = currentIndex + maxSize;

    if (endIndex < text.length) {
      // 1. Buscamos el último signo de puntuación que cierra una frase
      // Usamos una expresión regular para buscar . ! o ? seguido de un espacio o fin de línea
      const segment = text.substring(currentIndex, endIndex);
      const lastSentenceEnd = segment.search(/[.!?](?!\d)[^.!?]*$/);

      if (lastSentenceEnd !== -1) {
        // Ajustamos el corte justo después del punto (lastSentenceEnd + 1)
        endIndex = currentIndex + lastSentenceEnd + 1;
      } else {
        // Si no hay puntos en todo el bloque (párrafo muy largo), 
        // recurrimos a buscar el último espacio para no cortar palabras
        const lastSpace = text.lastIndexOf(" ", endIndex);
        if (lastSpace > currentIndex) {
          endIndex = lastSpace;
        }
      }
    }

    const chunk = text.substring(currentIndex, endIndex).trim();
    if (chunk.length > 0) {
      chunks.push(chunk);
    }

    // 2. Aplicamos el solapamiento
    currentIndex = endIndex - overlap;

    // 3. Control de seguridad para evitar solapamientos infinitos
    if (currentIndex <= 0 && chunks.length > 0) currentIndex = endIndex;
    if (currentIndex >= text.length - 10) break;
  }

  return chunks;
}