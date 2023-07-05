def get_db_uri(db_name: str):

    if not db_name or not db_name.strip():
        raise ValueError("It is necessary to provide db name")

    return f"sqlite:///{db_name}?check_same_thread=False&charset=utf8"
