"""Categorical text transformers."""

import re

from parsers.base.core import BaseCategoricalTextExtractor


class GenderExtractor(BaseCategoricalTextExtractor):
    """Transformer that extracts gender information from a column.

    This transformer searches for gender-specific keywords in each text entry
    and returns a standardized label ('male' or 'female') based on pre-defined
    regular expression patterns.
    """

    PATTERNS: dict[str, re.Pattern] = {
        "male": re.compile(r"\b(мужчина|male)\b", re.IGNORECASE),
        "female": re.compile(r"\b(женщина|female)\b", re.IGNORECASE),
    }


class WorkPositionCategorizer(BaseCategoricalTextExtractor):
    """Transformer to extract the main work position category from a column.

    This transformer searches for job-related keywords in each text entry and
    maps them to a predefined set of standardized categories, such as 'programmer',
    'qa', 'manager', 'designer', etc.
    """

    PATTERNS: dict[str, re.Pattern] = {
        "programmer": re.compile(
            r"программист|разработчик|developer|software engineer|full[-\s]?stack|frontend|backend|HTML|Java|Python",
            re.IGNORECASE,
        ),
        "qa": re.compile(r"test|тестировщик|QA|quality assurance|automation|devops", re.IGNORECASE),
        "sysadmin": re.compile(r"админ|administrator|network engineer", re.IGNORECASE),
        "engineer": re.compile(
            r"инженер|engineering|технический специалист|техник|engineer|technical specialist",
            re.IGNORECASE,
        ),
        "operator": re.compile(
            r"поддержка|support|оператор|operator|специалист|specialist|мастер|helpdesk",
            re.IGNORECASE,
        ),
        "manager": re.compile(
            r"менеджер|manager|руководитель|начальник|директор|управляющий|владелец|lead|head|coordinator|director|CEO|CTO|executive|product owner",
            re.IGNORECASE,
        ),
        "analyst": re.compile(r"аналитик|analyst|data scientist|данные", re.IGNORECASE),
        "designer": re.compile(r"дизайн|web|designer|UX|UI|2d|3d|graphic", re.IGNORECASE),
        "marketer": re.compile(
            r"marketing|маркетолог|SMM|SEO|digital|content|brand", re.IGNORECASE
        ),
        "salesman": re.compile(
            r"продавец|консультант|sales|retail|account manager|кассир", re.IGNORECASE
        ),
        "installer": re.compile(r"монтажник|installer", re.IGNORECASE),
    }


class CityCategorizer(BaseCategoricalTextExtractor):
    """Transformer to categorize cities from a column.

    This transformer analyzes each text entry and maps it to a broad city category
    based on predefined patterns.
    """

    PATTERNS: dict[str, re.Pattern] = {
        # --- Explicit cities ---
        "moscow": re.compile(r"\bмосква\b|\bmoscow\b", re.IGNORECASE),
        "spb": re.compile(r"\bсанкт[-\s]?петербург\b|\bспб\b|\bsaint petersburg\b", re.IGNORECASE),
        # --- Large cities mapped to 'big' ---
        "big": re.compile(
            r"\b(екатеринбург|новосибирск|казань|нижний\s?новгород|челябинск|самара|омск|"
            r"ростов-на-дону|уфа|красноярск|пермь|воронеж|волгоград|алматы|тюмень|ярославль|саратов)\b",
            re.IGNORECASE,
        ),
    }


class RelocationReadinessExtractor(BaseCategoricalTextExtractor):
    """Transformer to extract relocation readiness from a column.

    This transformer analyzes each text entry to determine whether the person
    is willing, unwilling, or wants to relocate.
    """

    PATTERNS: dict[str, re.Pattern] = {
        "no": re.compile(r"не готов[а]? к переезду|not willing to relocate", re.IGNORECASE),
        "yes": re.compile(r"готов[а]? к переезду|willing to relocate", re.IGNORECASE),
        "want": re.compile(r"хочу переехать|want to relocate", re.IGNORECASE),
    }


class BusinessTripReadinessExtractor(BaseCategoricalTextExtractor):
    """Transformer to extract business trip readiness from a column.

    This transformer analyzes each text entry to determine whether the person
    is willing, unwilling, or willing only to occasional business trips.
    """

    PATTERNS: dict[str, re.Pattern] = {
        "no": re.compile(
            r"не готов[а]? к командировкам|not prepared for business trips", re.IGNORECASE
        ),
        "yes": re.compile(r"готов[а]? к командировкам|prepared for business trips", re.IGNORECASE),
        "rarely": re.compile(
            r"готов[а]? к редким командировкам|prepared for occasional business trips",
            re.IGNORECASE,
        ),
    }


class EducationLevelExtractor(BaseCategoricalTextExtractor):
    """Transformer to extract the highest education level from a column.

    Uses a hierarchy of education levels to return the first match:
        'higher' > 'special' > 'secondary' > 'incomplete'.
    """

    PATTERNS: dict[str, re.Pattern] = {
        # --- Высшее образование ---
        "higher": re.compile(
            r"\b(?:высшее|бакалавр|магистр|специалист|higher)\b(?!.*неоконченное)",
            re.IGNORECASE,
        ),
        # --- Среднее специальное / колледж / техникум ---
        "special": re.compile(
            r"\b(?:техникум|колледж|college|училище|среднее специальное|special)\b",
            re.IGNORECASE,
        ),
        # --- Среднее образование ---
        "secondary": re.compile(r"\b(?:среднее|secondary)\b", re.IGNORECASE),
        # --- Неоконченное высшее ---
        "incomplete": re.compile(
            r"\b(?:неоконченное высшее|неоконченное обучение|incomplete)\b",
            re.IGNORECASE,
        ),
    }


class CarOwnershipExtractor(BaseCategoricalTextExtractor):
    """Transformer to determine whether a person owns a car from a column.

    This transformer searches for keywords in each text entry that indicate
    car ownership and maps them to standardized categories ('yes', 'no').
    """

    PATTERNS: dict[str, re.Pattern] = {
        "yes": re.compile(r"имеет|есть|да|own|has", re.IGNORECASE),
        "no": re.compile(r"нет|no|none|not own", re.IGNORECASE),
    }
