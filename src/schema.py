class OutputSchema:

    def __init__(self, pydantic_schema, schema_name: str):
        self._pydantic_schema = pydantic_schema
        self._schema_name = schema_name
        self._json_schema = self._get_json_schema_from_pydantic(pydantic_schema, schema_name)

    def __call__(self, *args, **kwargs):
        return self._pydantic_schema(*args, **kwargs)

    @property
    def tool(self):
        return self._json_schema

    @property
    def tool_choice(self):
        return {
            'type': 'function',
            'function': {'name': self._schema_name}
        }

    @staticmethod
    def _get_json_schema_from_pydantic(pydantic_schema, schema_name: str):
        properties = {}
        required = []
        for field_name, field_info in sorted(pydantic_schema.model_fields.items(), key=lambda x: x[0]):
            dtype = field_info.annotation
            if dtype != bool:
                raise NotImplementedError(f'Only bool is supported, got {dtype}')
            required.append(field_name)
            properties[field_name] = {'type': 'boolean'}

        schema = {
            'type': 'function',
            'function': {
                'name': schema_name,
                'parameters': {
                    'type': 'object',
                    'properties': properties,
                    'required': required
                }
            }
        }
        return schema
