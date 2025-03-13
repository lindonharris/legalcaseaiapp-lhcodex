# Define multiple sets of instruction templates i the dictionaries below

INSTRUCTION_TEMPLATES = {
################# PODCAST ##################
    "podcast": {
        "intro": """Your task is to take the input text provided and turn it into an lively, engaging, informative 
        podcast dialogue, in the style of NPR. The input text may be messy or unstructured, as it could come 
        from a variety of sources like PDFs or web pages.

        Don't worry about the formatting issues or any irrelevant information; your goal is to extract the 
        key points, identify definitions, and interesting facts that could be discussed in a podcast.

        Define all terms used carefully for a broad audience of listeners.""",

        "text_instructions": """First, carefully read through the input text and identify the main topics, key points, and 
        any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that 
        would be suitable for a high-quality presentation.""",
        
        "scratch_pad": """Brainstorm creative ways to discuss the main topics and key points you identified in the input text. 
        Consider using analogies, examples, storytelling techniques, or hypothetical scenarios to make the content more 
        relatable and engaging for listeners.

        Keep in mind that your podcast should be accessible to a general audience, so avoid using too much jargon or 
        assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

        Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that 
        could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free 
        to be creative in your approach.

        Define all terms used clearly and spend effort to explain the background.

        Write your brainstorming ideas and a rough outline for the podcast dialogue here. Be sure to note the key 
        insights and takeaways you want to reiterate at the end.

        Make sure to make it fun and exciting.""",

        "prelude": """Now that you have brainstormed ideas and created a rough outline, it's time to write the actual 
        podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the 
        best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.""",
        
        "dialog": """Write a very long, engaging, informative podcast dialogue here, based on the key points and creative 
        ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary 
        context or explanations to make the content accessible to a general audience.

        Never use made-up names for the hosts and guests, but make it an engaging and immersive experience for listeners. 
        Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be 
        directly converted into audio.

        Make the dialogue as long and detailed as possible, while still staying on topic and maintaining an engaging flow. 
        Aim to use your full output capacity to create the longest podcast episode you can, while still communicating the 
        key information from the input text in an entertaining way.

        At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and 
        takeaways from their discussion. This should flow organically from the conversation, reiterating the key 
        points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce 
        the central ideas one last time before signing off.

        The podcast should have around 20000 words.""",
    },

################# MATERIAL DISCOVERY SUMMARY ##################
    "SciAgents material discovery summary": {
        "intro": """Your task is to take the input text provided and turn it into a lively, engaging conversation between a professor and a student in a panel discussion that describes a new material. The professor acts like Richard Feynman, but you never mention the name.

The input text is the result of a design developed by SciAgents, an AI tool for scientific discovery that has come up with a detailed materials design.

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that could be discussed in a podcast.

Define all terms used carefully for a broad audience of listeners.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for a high quality presentation.",
        "scratch_pad": """Brainstorm creative ways to discuss the main topics and key points you identified in the material design summary, especially paying attention to design features developed by SciAgents. Consider using analogies, examples, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your description should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.

Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the podcast dialogue here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it fun and exciting. You never refer to the podcast, you just discuss the discovery and you focus on the new material design only.
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a very long, engaging, informative dialogue here, based on the key points and creative ideas you came up with during the brainstorming session. The presentation must focus on the novel aspects of the material design, behavior, and all related aspects.

Use a conversational tone and include any necessary context or explanations to make the content accessible to a general audience, but make it detailed, logical, and technical so that it has all necessary aspects for listeners to understand the material and its unexpected properties.

Remember, this describes a design developed by SciAgents, and this must be explicitly stated for the listeners.

Never use made-up names for the hosts and guests, but make it an engaging and immersive experience for listeners. Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

Make the dialogue as long and detailed as possible with great scientific depth, while still staying on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest podcast episode you can, while still communicating the key information from the input text in an entertaining way.

At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and takeaways from their discussion. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas one last time before signing off.

The conversation should have around 20000 words.
"""
    },

################# LECTURE ##################
    "lecture": {
        "intro": """You are Professor Richard Feynman. Your task is to develop a script for a lecture. You never mention your name.

The material covered in the lecture is based on the provided text. 

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be covered in the lecture. 

Define all terms used carefully for a broad audience of students.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for a high quality presentation.",
        "scratch_pad": """
Brainstorm creative ways to discuss the main topics and key points you identified in the input text. Consider using analogies, examples, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your lecture should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.

Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the lecture here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it fun and exciting. 
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a very long, engaging, informative script here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to the students.

Include clear definitions and terms, and examples. 

Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

There is only one speaker, you, the professor. Stay on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest lecture you can, while still communicating the key information from the input text in an engaging way.

At the end of the lecture, naturally summarize the main insights and takeaways from the lecture. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. 

Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas covered in this lecture one last time before class is over. 

The lecture should have around 20000 words.
""",
    },

################# SUMMARY ##################
    "summary": {
        "intro": """Your task is to develop a summary of a paper. You never mention your name.

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be summarized.

Define all terms used carefully for a broad audience.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and key facts. Think about how you could present this information in an accurate summary.",
        "scratch_pad": """Brainstorm creative ways to present the main topics and key points you identified in the input text. Consider using analogies, examples, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your summary should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms. Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the summary here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it engaging and exciting. 
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it is time to write the actual summary. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a a script here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to the the audience.

Start your script by stating that this is a summary, referencing the title or headings in the input text. If the input text has no title, come up with a succinct summary of what is covered to open.

Include clear definitions and terms, and examples, of all key issues. 

Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

There is only one speaker, you. Stay on topic and maintaining an engaging flow. 

Naturally summarize the main insights and takeaways from the summary. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. 

The summary should have around 1024 words.
""",
    },

################# SHORT SUMMARY ##################
    "short summary": {
        "intro": """Your task is to develop a summary of a paper. You never mention your name.

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be summarized.

Define all terms used carefully for a broad audience.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and key facts. Think about how you could present this information in an accurate summary.",
        "scratch_pad": """Brainstorm creative ways to present the main topics and key points you identified in the input text. Consider using analogies, examples, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your summary should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms. Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the summary here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it engaging and exciting. 
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it is time to write the actual summary. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a script here, based on the key points and creative ideas you came up with during the brainstorming session. Keep it concise, and use a conversational tone and include any necessary context or explanations to make the content accessible to the the audience.

Start your script by stating that this is a summary, referencing the title or headings in the input text. If the input text has no title, come up with a succinct summary of what is covered to open.

Include clear definitions and terms, and examples, of all key issues. 

Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

There is only one speaker, you. Stay on topic and maintaining an engaging flow. 

Naturally summarize the main insights and takeaways from the short summary. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. 

The summary should have around 256 words.
""",
    },

}


INSTRUCTION_TEMPLATES_ESP = {
################# PODCAST SPANISH ##################
    "podcast (Spanish)": {
        "intro": """Tu tarea es tomar el texto de entrada proporcionado y convertirlo en un diálogo de podcast animado, 
        atractivo e informativo, al estilo de NPR. El texto de entrada puede estar desordenado o poco estructurado, ya que 
        podría provenir de diversas fuentes como archivos PDF o páginas web.

        No te preocupes por los problemas de formato o por la información irrelevante; tu objetivo es extraer los puntos clave, 
        identificar definiciones y hechos interesantes que podrían discutirse en un podcast.

        Definir cuidadosamente todos los términos utilizados para una audiencia amplia.""",

        "text_instructions": """Primero, lee detenidamente el texto de entrada e identifica los temas principales, 
        los puntos clave y cualquier hecho o anécdota interesante. Piensa en cómo podrías presentar esta información 
        de una manera divertida y atractiva, adecuada para una presentación de alta calidad.""",
    
        "scratch_pad": """Piensa de manera creativa sobre cómo discutir los temas principales y los puntos clave 
        que has identificado en el texto de entrada. Considera usar analogías, ejemplos, técnicas narrativas o 
        escenarios hipotéticos para hacer que el contenido sea más comprensible y atractivo para los oyentes.

        Ten en cuenta que tu podcast debe ser accesible para una audiencia general, así que evita usar demasiado 
        jerga técnica o asumir que la audiencia tiene conocimientos previos del tema. Si es necesario, piensa en 
        formas de explicar brevemente cualquier concepto complejo en términos sencillos.

        Usa tu imaginación para llenar los vacíos en el texto de entrada o para formular preguntas provocadoras 
        que podrían explorarse en el podcast. El objetivo es crear un diálogo informativo y entretenido, por lo que 
        puedes ser creativo en tu enfoque.

        Define claramente todos los términos utilizados y asegúrate de explicar el trasfondo.

        Escribe tus ideas de brainstorming y un esquema general del diálogo del podcast aquí. Asegúrate de anotar 
        los puntos clave y las conclusiones que deseas reiterar al final.

        Asegúrate de que sea divertido y emocionante.""",

        "prelude": """Ahora que has realizado una lluvia de ideas y has creado un esquema general, es hora de escribir 
        el diálogo real del podcast. Apunta a un flujo natural y conversacional entre el presentador y cualquier 
        invitado. Incorpora las mejores ideas de tu sesión de lluvia de ideas y asegúrate de explicar cualquier tema 
        complejo de una manera fácil de entender.""",
        
        "dialog": """Escribe aquí un diálogo de podcast muy largo, atractivo e informativo, basado en los puntos clave y 
        las ideas creativas que se te ocurrieron durante la sesión de brainstorming. Usa un tono conversacional e incluye 
        el contexto o las explicaciones necesarias para que el contenido sea accesible a una audiencia general.

        Nunca uses nombres inventados para los presentadores e invitados, pero haz que sea una experiencia 
        atractiva e inmersiva para los oyentes. No incluyas ningún marcador de posición entre corchetes 
        como [Presentador] o [Invitado]. Diseña tu salida para que sea leída en voz alta, ya que se 
        convertirá directamente en audio.

        Haz el diálogo lo más largo y detallado posible, manteniéndote en el tema y asegurando un flujo atractivo. 
        Apunta a utilizar toda tu capacidad de salida para crear el episodio de podcast más largo posible, mientras 
        comunicas la información clave del texto de entrada de una manera entretenida.

        Al final del diálogo, el presentador y los invitados deben resumir naturalmente las principales ideas y 
        conclusiones de su conversación. Esto debe fluir orgánicamente desde la conversación, reiterando los 
        puntos clave de manera casual y conversacional. Evita que suene como un resumen obvio: el objetivo es 
        reforzar las ideas centrales una última vez antes de finalizar.

        El podcast debe tener alrededor de 20000 palabras.""",
    }, 
}
