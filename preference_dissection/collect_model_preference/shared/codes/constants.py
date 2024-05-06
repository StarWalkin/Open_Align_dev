scenario_group = {
    "Exam Questions": ["math_reasoning", "solving_exam_question_with_math", "solving_exam_question_without_math", ],
    # checked!
    "Code": ["code_simplification",
             "code_generation",
             "explaining_code",
             "code_correction_rewriting",
             "code_to_code_translation",
             ],  # checked!
    "Creative Writing": ["writing_song_lyrics", "writing_social_media_post", "writing_blog_post",
                         "writing_personal_essay",
                         "creative_writing", "writing_advertisement", "writing_marketing_materials",
                         "writing_presentation_script",
                         "counterfactual", ],  # checked!
    "Functional Writing": [
        "writing_product_description",
        "writing_job_application",
        "writing_news_article",
        "writing_biography",
        "writing_email",
        "writing_legal_document",
        "writing_technical_document",
        "writing_scientific_paper",
        "functional_writing",
        "writing_cooking_recipe",
    ],  # checked!
    "Communication": ["value_judgement", "chitchat"],  # checked!
    "Knowledge-aware": ["open_question", "explaining_general", "verifying_fact", ],  # checked!
    "Advice": ["asking_how_to_question", "seeking_advice", ],  # checked!
    "Daily Tasks": ["analyzing_general", "roleplay", "planning", "recommendation", "brainstorming"],  # checked!
    "NLP Tasks": [
        "ranking",
        "text_to_text_translation",
        "classification_identification",
        "title_generation",
        "question_generation",
        "reading_comprehension",
        "keywords_extraction",
        "information_extraction",
        "topic_modeling",
        "data_analysis",
        "post_summarization", "text_summarization", "note_summarization",
        "text_simplification",
        "language_polishing",
        "instructional_rewriting",
        "text_correction",
        "paraphrasing",
    ],  # checked!
    "Others": ["default"],  # checked!
    "Unsafe Queries": ["rejecting"],  # -> use `oaimoderation` and `toxic-check` to choose
}

reversed_scenario_group = {
    vv: k for k, v in scenario_group.items() for vv in v
}
