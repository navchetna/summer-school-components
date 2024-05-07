from main import Summarizer


if __name__ == "__main__":
    client = Summarizer()
    # ARTICLE_TO_SUMMARIZE = (
    # "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    # "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    # "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
    # )
    ARTICLE_TO_SUMMARIZE = """Living life as a single individual can be a unique and fulfilling experience, offering a multitude of opportunities for personal growth, self-discovery, and independence. Embracing solitude can provide a sense of freedom and autonomy, allowing one to focus on their own needs, desires, and aspirations without the constraints of a romantic relationship. Single life offers the chance to cultivate a strong sense of self-awareness and self-reliance, fostering a deeper understanding of one's own values, beliefs, and goals. It provides the space and time to explore individual interests, hobbies, and passions, leading to a greater sense of fulfillment and purpose.
In the realm of personal development, being single can be a catalyst for self-improvement and growth. It presents an opportunity to engage in introspection, reflection, and introspective practices that can lead to enhanced emotional intelligence, resilience, and self-confidence. Without the distractions or compromises that often come with a partnership, individuals can fully dedicate themselves to personal development, pursuing education, career advancement, or creative endeavors with unwavering focus and determination.
Moreover, single life offers a unique perspective on relationships and interpersonal dynamics. By experiencing life independently, individuals can gain a deeper understanding of their own needs and boundaries, leading to healthier and more fulfilling relationships in the future. It allows for the exploration of different types of connections, friendships, and social interactions, fostering a diverse and enriching social network that can provide support, companionship, and camaraderie.
From a practical standpoint, being single can also bring financial stability, flexibility, and freedom. Without the financial responsibilities or constraints of a shared household, individuals have the opportunity to prioritize their own financial goals, savings, and investments. This financial independence can lead to a greater sense of security and empowerment, enabling individuals to make decisions based on their own needs and aspirations rather than those of a partner.
In terms of personal well-being, single life can promote a healthy lifestyle, self-care, and overall well-being. It allows individuals to prioritize their physical, mental, and emotional health, engaging in activities that promote relaxation, stress relief, and overall wellness. Whether it's practicing mindfulness, pursuing fitness goals, or simply enjoying moments of solitude and reflection, single life offers the freedom to prioritize self-care and personal well-being.
In conclusion, life as a single individual can be a rich and rewarding experience, offering a multitude of opportunities for personal growth, self-discovery, and fulfillment. Embracing solitude can lead to enhanced self-awareness, independence, and resilience, fostering a deeper understanding of oneself and the world around us. By embracing the unique opportunities and challenges of single life, individuals can cultivate a sense of empowerment, purpose, and contentment that can enrich every aspect of their lives"""


    print(client.summary_generator(ARTICLE_TO_SUMMARIZE))
    

