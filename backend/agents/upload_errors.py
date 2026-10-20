"""Errori classificati e messaggi stabili per l'ingestione dei PDF."""


class UploadError(ValueError):
    def __init__(self, code: str, message: str, action: str):
        self.code = code
        self.action = action
        super().__init__(f"{message} Azione suggerita: {action}")


MESSAGES = {
    "non_pdf": ("Il file non è un PDF valido o è stato rinominato.", "Esporta nuovamente il documento in PDF."),
    "corrotto": ("Il PDF è corrotto o troncato.", "Controlla il file e ricarica una copia completa."),
    "password": ("Il PDF è protetto da password.", "Rimuovi la protezione e ricarica il PDF."),
    "scansionato": ("L'OCR non ha trovato testo leggibile nel PDF scansionato.", "Carica una scansione più nitida o un PDF con testo selezionabile."),
    "ocr_fallito": ("Non è stato possibile leggere una pagina scansionata con OCR.", "Controlla il PDF e riprova."),
    "troppo_grande": ("Il file supera la dimensione massima consentita.", "Riduci o comprimi il PDF prima di ricaricarlo."),
    "zero_chunk": ("Il PDF non contiene chunk utili dopo il filtro.", "Ricarica un PDF con testo più esteso o leggibile."),
    "testo_illeggibile": ("Il testo del PDF non è leggibile.", "Rigenera il PDF con codifica Unicode o applica OCR."),
}


def upload_error(code: str) -> UploadError:
    message, action = MESSAGES[code]
    return UploadError(code, message, action)
