import { GoogleGenAI, GenerateContentResponse, Chat } from "@google/genai";

const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
const ai = new GoogleGenAI({ apiKey: apiKey });


export interface MarketingAnalysis {
  pros: string[];
  cons: string[];
  swot: {
    strengths: string[];
    weaknesses: string[];
    opportunities: string[];
    threats: string[];
  };
  businessPlan: string;
}

export interface TeamMemberRecommendation {
  name: string;
  role: string;
  tasks: string;
  compensation: number;
  compensationType: 'monthly' | 'hourly' | 'fixed';
}

export async function analyzeMarketingIdea(idea: string, category: string): Promise<MarketingAnalysis> {
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: `Analizza la seguente idea di marketing per un progetto di tipo "${category}": "${idea}". 
    Fornisci un'analisi SWOT dettagliata, una lista di pro e contro e una breve struttura di business plan.
    Rispondi in ITALIANO.
    Restituisci la risposta in formato JSON con la seguente struttura:
    {
      "pros": ["string"],
      "cons": ["string"],
      "swot": {
        "strengths": ["string"],
        "weaknesses": ["string"],
        "opportunities": ["string"],
        "threats": ["string"]
      },
      "businessPlan": "stringa in formato markdown"
    }`,
    config: {
      responseMimeType: "application/json",
    },
  });

  try {
    return JSON.parse(response.text || "{}");
  } catch (e) {
    console.error("Failed to parse analysis JSON", e);
    throw new Error("Errore nell'analisi dell'idea");
  }
}

export async function analyzeCompetitors(idea: string, scope: string, category: string, urls: string[] = []): Promise<string> {
  const urlContext = urls.length > 0 ? `\nInoltre, analizza in modo approfondito le strategie online di questi siti web dei competitor:\n${urls.map(u => `- ${u}`).join('\n')}` : '';

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: `Sei un esperto analista di mercato e stratega aziendale.
    L'utente sta avviando un progetto con queste caratteristiche:
    - Idea: "${idea}"
    - Categoria: "${category}"
    - Localizzazione / Ambito / Tipo di attività: "${scope}"${urlContext}
    
    Effettua un'analisi dei competitor dettagliata. Utilizza dati demografici e di mercato per:
    1. Identificare i principali concorrenti (diretti e indiretti) nella zona specificata o nel settore a livello nazionale.
    2. Analizzare i loro punti di forza e di debolezza.
    3. Suggerire strategie pratiche e mirate per differenziarsi e ottenere un vantaggio competitivo.
    ${urls.length > 0 ? '4. Fornire un\'analisi approfondita delle strategie online dei competitor forniti tramite URL.' : ''}
    
    Rispondi in ITALIANO usando la formattazione Markdown per una facile lettura.`,
    config: {
      tools: [{ googleSearch: {} }],
    }
  });
  return response.text || "Nessuna analisi generata.";
}

export async function analyzeLocation(location: string, idea: string, category: string, radius: number, businessType: string, lat?: number, lng?: number) {
  try {
    const query = businessType === 'all' 
      ? `Identifica le principali attività commerciali (bar, ristoranti, meccanici, negozi, centri sportivi) entro un raggio di ${radius}km da ${location} per supportare un progetto di tipo "${category}" con l'idea: "${idea}".`
      : `Identifica i principali ${businessType} entro un raggio di ${radius}km da ${location} per supportare un progetto di tipo "${category}" con l'idea: "${idea}".`;

    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: `${query} 
      Seleziona esclusivamente le attività con la MAGGIORE PROBABILITÀ di essere interessate a una sponsorizzazione basandoti sulla pertinenza con l'idea.
      Fornisci un elenco puntato dei potenziali sponsor, indicando per ognuno il motivo specifico per cui dovrebbero partecipare.
      Rispondi in ITALIANO. Sii concreto e diretto.`,
      config: {
        tools: [{ googleMaps: {} }],
        toolConfig: {
          retrievalConfig: {
            latLng: lat && lng ? { latitude: lat, longitude: lng } : undefined
          }
        }
      },
    });

    if (!response.text) {
      throw new Error("Nessuna risposta ricevuta dal servizio di mappe.");
    }

    return {
      text: response.text,
      sources: response.candidates?.[0]?.groundingMetadata?.groundingChunks || []
    };
  } catch (error: any) {
    console.error("Errore in analyzeLocation:", error);
    throw new Error(error.message || "Errore durante la ricerca su Google Maps.");
  }
}

export async function findSponsorsAI(idea: string, category: string, location: string) {
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: `Analizza l'idea: "${idea}" (categoria: ${category}) a "${location}".
      Identifica i 3-5 profili di sponsor ideali (categorie di aziende).
      Per ogni categoria fornisci:
      1. MOTIVAZIONE: Perché è il partner perfetto.
      2. STRATEGIA: Consiglio esperto su come approcciarli (breve e schematico).
      Usa un formato elenco puntato molto sintetico e professionale.
      Rispondi in ITALIANO.`,
    });

    return response.text || "Nessuna analisi generata.";
  } catch (error: any) {
    console.error("Errore in findSponsorsAI:", error);
    throw new Error(error.message || "Errore durante l'analisi degli sponsor AI.");
  }
}

export async function recommendTeam(idea: string, category: string, expectedRevenue: number): Promise<TeamMemberRecommendation[]> {
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: `Sei un Project Manager esperto di livello mondiale.
    L'utente sta pianificando un progetto di tipo "${category}" con la seguente idea: "${idea}".
    Le entrate previste (budget/revenue) per questo progetto sono stimate a €${expectedRevenue}.
    
    Il tuo compito è consigliare una struttura del team ottimale e realistica per questo progetto, basandoti sulle entrate previste. 
    Non superare il budget ragionevole per le risorse umane (di solito una percentuale delle entrate).
    Per ogni membro del team, fornisci:
    - name: Un nome fittizio o generico (es. "Risorsa 1", "Sviluppatore Senior", ecc.)
    - role: Il ruolo specifico nel progetto.
    - tasks: I compiti principali che dovrà svolgere.
    - compensation: La retribuzione consigliata in Euro (numero).
    - compensationType: Il tipo di retribuzione ("monthly", "hourly", o "fixed").
    
    Rispondi in ITALIANO.
    Restituisci la risposta in formato JSON con la seguente struttura:
    {
      "team": [
        {
          "name": "string",
          "role": "string",
          "tasks": "string",
          "compensation": number,
          "compensationType": "monthly" | "hourly" | "fixed"
        }
      ]
    }`,
    config: {
      responseMimeType: "application/json",
    },
  });

  try {
    const data = JSON.parse(response.text || "{}");
    return data.team || [];
  } catch (e) {
    console.error("Failed to parse team JSON", e);
    throw new Error("Errore nella generazione del team consigliato");
  }
}

export function createMarketingChat(): Chat {
  return ai.chats.create({
    model: "gemini-2.5-flash",
    config: {
      systemInstruction: `Sei un esperto di marketing di livello mondiale con una profonda conoscenza del mercato italiano. 
      Fornisci consulenza strategica, idee creative e suggerimenti pratici. 
      È FONDAMENTALE che i tuoi consigli tengano conto della normativa italiana vigente, inclusi ma non limitati a:
      - GDPR e normativa sulla privacy italiana (Garante Privacy).
      - Codice del Consumo italiano.
      - Normative sulla pubblicità e sull'e-commerce in Italia.
      - Regole dell'AGCM (Autorità Garante della Concorrenza e del Mercato).
      Rispondi sempre in ITALIANO. Sii professionale, perspicace e incoraggiante.`,
    },
  });
}

export function createProjectManagerChat(idea?: string, expectedRevenue?: number): Chat {
  const context = idea ? `
  CONTESTO DEL PROGETTO:
  - Idea: "${idea}"
  - Entrate Previste (Budget stimato): €${expectedRevenue || 0}
  ` : '';

  return ai.chats.create({
    model: "gemini-2.5-flash",
    config: {
      systemInstruction: `Sei un Project Manager esperto di livello mondiale.
      Il tuo compito è aiutare l'utente a pianificare e realizzare il suo progetto con un approccio TECNICO e legato al Project Management (non al marketing).
      ${context}
      Devi fornire stime su:
      - Numero di persone necessarie e i loro ruoli.
      - Retribuzione consigliata in base alle ENTRATE PREVISTE (budget). Se le entrate sono basse, consiglia un team snello o retribuzioni a percentuale/fisse. Se sono alte, struttura un team più completo.
      - Tempistiche e fasi del progetto (Milestones, diagrammi di Gantt testuali).
      - Strumenti, metodologie (Agile, Scrum, Waterfall) e risorse necessarie.
      
      Sii strutturato, professionale, chiaro e pragmatico. Usa elenchi puntati e tabelle markdown quando utile.
      Rispondi sempre in ITALIANO.`,
    },
  });
}

export async function suggestCompensation(idea: string, role: string, expectedRevenue: number): Promise<{ amount: number, type: 'monthly' | 'hourly' | 'fixed' }> {
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: `Sei un esperto HR e Project Manager.
    L'utente sta creando un team per la seguente idea di business: "${idea}".
    Le entrate previste (budget) sono: €${expectedRevenue}.
    L'utente vuole aggiungere un membro con il ruolo: "${role}".
    
    Suggerisci una retribuzione realistica per questo ruolo, tenendo conto del budget disponibile.
    Se il budget è basso, suggerisci una retribuzione oraria o fissa più bassa.
    Se il budget è alto, suggerisci una retribuzione mensile di mercato.
    
    Rispondi SOLO con un oggetto JSON con questo formato esatto:
    {
      "amount": numero (la cifra suggerita),
      "type": "monthly" | "hourly" | "fixed" (il tipo di retribuzione)
    }`,
    config: {
      responseMimeType: "application/json",
    },
  });

  try {
    const data = JSON.parse(response.text || "{}");
    return {
      amount: data.amount || 0,
      type: data.type || 'monthly'
    };
  } catch (e) {
    console.error("Failed to parse compensation JSON", e);
    return { amount: 0, type: 'monthly' };
  }
}

export function createAccountantChat(): Chat {
  return ai.chats.create({
    model: "gemini-2.5-flash",
    config: {
      systemInstruction: `Sei un Commercialista esperto iscritto all'Albo in Italia, specializzato in consulenza per startup, piccole medie imprese (PMI) e liberi professionisti.
      La tua missione è fornire informazioni precise, aggiornate e professionali sulla legge italiana, con particolare attenzione a:
      - Regime Forfettario vs Regime Ordinario.
      - Costituzione di società (SRL, SRLS, SAS, SNC).
      - Adempimenti fiscali e scadenze (IVA, IRPEF, IRES, IRAP).
      - Contributi previdenziali (INPS, Casse Professionali).
      - Agevolazioni fiscali, crediti d'imposta e incentivi per l'imprenditoria (es. Resto al Sud, Nuove Imprese a Tasso Zero).
      - Fatturazione elettronica.
      - Diritto del lavoro e contrattualistica base.
      
      IMPORTANTE:
      1. Cita sempre, quando possibile, le leggi o i decreti di riferimento (es. TUIR, Legge di Bilancio).
      2. Sii estremamente preciso ma usa un linguaggio comprensibile anche a chi non è esperto.
      3. Ricorda sempre all'utente che i tuoi consigli sono a scopo informativo e che per operazioni complesse è sempre necessario il supporto diretto di un professionista abilitato che analizzi il caso specifico.
      4. Rispondi sempre in ITALIANO.`,
    },
  });
}
