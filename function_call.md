# Registro Automático de Funções C++ como Tools

## 1. Objetivo

O objetivo desta proposta é simplificar o registro de **tools** no `MiniCPM5.cpp`.

Em vez de criar uma nova classe derivada de `Tool` para cada operação, a ideia é permitir que uma função C++ comum seja registrada diretamente:

```cpp
int add(int a, int b)
{
    return a + b;
}

registry.add_function(
    "add",
    "Adds two integers",
    add,
    {
        {"a", "First number"},
        {"b", "Second number"}
    }
);
```

O sistema deve ser capaz de descobrir automaticamente que:

```cpp
int add(int a, int b);
```

possui:

- retorno do tipo `int`;
- primeiro argumento do tipo `int`;
- segundo argumento do tipo `int`.

Com isso, o framework pode gerar automaticamente o JSON Schema da tool e converter os argumentos recebidos do modelo para os tipos C++ corretos.

A principal vantagem é reduzir o boilerplate necessário para adicionar novas funções ao sistema.

---

# 2. Problema atual

Em uma implementação tradicional baseada em classes, cada tool precisa fornecer explicitamente:

- nome;
- descrição;
- definição dos argumentos;
- JSON Schema;
- lógica de conversão dos argumentos;
- implementação de `execute()`.

Conceitualmente:

```cpp
class AddTool : public Tool
{
public:
    std::string name() const override
    {
        return "add";
    }

    std::string definition() const override
    {
        // construir JSON manualmente
    }

    std::string execute(
        const std::vector<std::pair<std::string, std::string>>& params
    ) override
    {
        int a = std::stoi(...);
        int b = std::stoi(...);

        return std::to_string(a + b);
    }
};
```

Para uma tool simples, há muito mais código de infraestrutura do que código funcional.

A proposta é transformar:

```cpp
int add(int a, int b)
{
    return a + b;
}
```

diretamente em uma tool.

---

# 3. Informações que o C++ consegue deduzir

Considere:

```cpp
int add(int a, double b, const std::string& text);
```

O tipo da função é:

```cpp
int (*)(int, double, const std::string&)
```

Em C++ é possível inspecionar esse tipo usando templates.

O compilador conhece:

```text
return type = int

arg 0 = int
arg 1 = double
arg 2 = const std::string&
```

Essas informações existem em **compile time**.

O framework pode usá-las para construir automaticamente:

```text
int         -> integer
double      -> number
std::string -> string
```

---

# 4. O que o C++ não consegue deduzir

Existe uma limitação importante.

Dada a função:

```cpp
int add(int a, int b);
```

os templates conseguem enxergar:

```text
int(int, int)
```

mas não conseguem recuperar de forma portável os nomes:

```text
a
b
```

Ou seja:

```text
TIPOS     -> podem ser deduzidos
NOMES     -> precisam ser informados
DESCRIÇÃO -> precisa ser informada
```

Por isso o registro ainda precisa fornecer metadados mínimos:

```cpp
registry.add_function(
    "add",
    "Adds two integers",
    add,
    {
        {"a", "First number"},
        {"b", "Second number"}
    }
);
```

O binder combina essas duas fontes de informação:

```text
Metadado           Assinatura C++

"a"                int
"b"                int
```

e gera a definição completa.

---

# 5. `function_traits`

O componente central da implementação é um mecanismo normalmente chamado de:

```text
function_traits
```

Ele permite decompor o tipo de uma função.

Primeiro declaramos o template genérico:

```cpp
template<typename T>
struct function_traits;
```

Em seguida fazemos uma especialização para ponteiros de função:

```cpp
template<typename R, typename... Args>
struct function_traits<R(*)(Args...)>
{
    using return_type = R;

    static constexpr std::size_t arity = sizeof...(Args);

    template<std::size_t I>
    using arg =
        typename std::tuple_element<
            I,
            std::tuple<Args...>
        >::type;
};
```

Aqui:

```cpp
R
```

representa o tipo de retorno.

E:

```cpp
Args...
```

representa todos os argumentos da função.

Para:

```cpp
int add(int a, double b);
```

o compilador instancia algo equivalente a:

```cpp
R = int

Args... = <
    int,
    double
>
```

Portanto:

```cpp
using traits = function_traits<decltype(&add)>;
```

permite consultar:

```cpp
traits::return_type
```

que corresponde a:

```cpp
int
```

e:

```cpp
traits::arg<0>
```

corresponde a:

```cpp
int
```

enquanto:

```cpp
traits::arg<1>
```

corresponde a:

```cpp
double
```

Também podemos obter a quantidade de argumentos:

```cpp
traits::arity
```

que neste caso será:

```text
2
```

---

# 6. Removendo `const`, referências e qualificadores

É comum encontrar funções como:

```cpp
void print_message(const std::string& text);
```

O argumento é tecnicamente:

```cpp
const std::string&
```

mas para fins de serialização queremos tratar esse tipo simplesmente como:

```cpp
std::string
```

Para isso podemos usar:

```cpp
std::decay_t<T>
```

Exemplo:

```cpp
template<typename T>
using clean_type = std::decay_t<T>;
```

Então:

```cpp
clean_type<const std::string&>
```

se transforma em:

```cpp
std::string
```

O mesmo vale para outros qualificadores.

---

# 7. Mapeamento C++ -> JSON Schema

Depois de descobrir o tipo do argumento, podemos convertê-lo para um tipo conhecido pelo JSON Schema.

Exemplo:

```cpp
template<typename T>
constexpr const char* json_type()
{
    using U = std::decay_t<T>;

    if constexpr (std::is_same_v<U, std::string>)
        return "string";

    else if constexpr (std::is_same_v<U, bool>)
        return "boolean";

    else if constexpr (std::is_integral_v<U>)
        return "integer";

    else if constexpr (std::is_floating_point_v<U>)
        return "number";

    else
        return "string";
}
```

Assim:

```cpp
json_type<int>()
```

retorna:

```text
integer
```

e:

```cpp
json_type<double>()
```

retorna:

```text
number
```

Tabela básica:

| C++ | JSON Schema |
|---|---|
| `int` | `integer` |
| `long` | `integer` |
| `unsigned int` | `integer` |
| `float` | `number` |
| `double` | `number` |
| `bool` | `boolean` |
| `std::string` | `string` |

Mais tipos podem ser adicionados futuramente.

---

# 8. Exemplo de geração automática do schema

Considere:

```cpp
double distance(double x, double y);
```

Registramos:

```cpp
registry.add_function(
    "distance",
    "Computes a distance",
    distance,
    {
        {"x", "X coordinate"},
        {"y", "Y coordinate"}
    }
);
```

O binder obtém:

```text
arg 0 -> double
arg 1 -> double
```

e combina com:

```text
arg 0 -> "x"
arg 1 -> "y"
```

O resultado pode ser:

```json
{
  "name": "distance",
  "description": "Computes a distance",
  "parameters": {
    "type": "object",
    "properties": {
      "x": {
        "type": "number",
        "description": "X coordinate"
      },
      "y": {
        "type": "number",
        "description": "Y coordinate"
      }
    },
    "required": [
      "x",
      "y"
    ]
  }
}
```

Nenhum tipo precisou ser informado manualmente.

---

# 9. Conversão dos argumentos recebidos

O modelo normalmente fornece os argumentos como texto ou JSON.

Por exemplo:

```json
{
  "a": "10",
  "b": "32"
}
```

Mas a função C++ espera:

```cpp
int add(int a, int b);
```

Portanto o binder precisa transformar:

```text
"10" -> int
"32" -> int
```

Podemos criar:

```cpp
template<typename T>
T parse_tool_value(const std::string& value);
```

E especializações:

```cpp
template<>
inline int parse_tool_value<int>(const std::string& value)
{
    return std::stoi(value);
}
```

Para `double`:

```cpp
template<>
inline double parse_tool_value<double>(const std::string& value)
{
    return std::stod(value);
}
```

Para `float`:

```cpp
template<>
inline float parse_tool_value<float>(const std::string& value)
{
    return std::stof(value);
}
```

Para `bool`:

```cpp
template<>
inline bool parse_tool_value<bool>(const std::string& value)
{
    return value == "true" ||
           value == "1";
}
```

Para string:

```cpp
template<>
inline std::string parse_tool_value<std::string>(
    const std::string& value)
{
    return value;
}
```

---

# 10. Como chamar automaticamente uma função

Agora temos dois componentes:

```text
function_traits
```

que descobre os tipos;

e:

```text
parse_tool_value<T>()
```

que converte strings para esses tipos.

O próximo passo é chamar a função automaticamente.

Considere:

```cpp
int add(int a, int b);
```

e os valores recebidos:

```cpp
values[0] = "10";
values[1] = "20";
```

Queremos gerar automaticamente algo equivalente a:

```cpp
add(
    parse_tool_value<int>(values[0]),
    parse_tool_value<int>(values[1])
);
```

Para isso usamos:

```cpp
std::index_sequence
```

---

# 11. `std::index_sequence`

Podemos escrever:

```cpp
template<typename F, std::size_t... I>
auto invoke_function(
    F& fn,
    const std::vector<std::string>& values,
    std::index_sequence<I...>)
{
    using traits = function_traits<F>;

    return fn(
        parse_tool_value<
            std::decay_t<
                typename traits::template arg<I>
            >
        >(values[I])...
    );
}
```

A parte:

```cpp
I...
```

é um **parameter pack**.

Para uma função com três argumentos:

```cpp
foo(int, float, std::string)
```

a sequência gerada será:

```text
0, 1, 2
```

A expansão:

```cpp
parse_tool_value<
    std::decay_t<typename traits::template arg<I>>
>(values[I])...
```

torna-se conceitualmente:

```cpp
parse_tool_value<int>(values[0]),
parse_tool_value<float>(values[1]),
parse_tool_value<std::string>(values[2])
```

Portanto o compilador gera algo equivalente a:

```cpp
foo(
    parse_tool_value<int>(values[0]),
    parse_tool_value<float>(values[1]),
    parse_tool_value<std::string>(values[2])
);
```

A quantidade de argumentos também é conhecida em compile time:

```cpp
traits::arity
```

Então podemos chamar:

```cpp
invoke_function(
    fn,
    values,
    std::make_index_sequence<traits::arity>{}
);
```

---

# 12. Fluxo completo

Para:

```cpp
int add(int a, int b)
{
    return a + b;
}
```

o processo completo seria:

```text
             C++ function
                  |
                  v
          int add(int, int)
                  |
                  v
           function_traits
                  |
         +--------+--------+
         |                 |
         v                 v
      arg<0>             arg<1>
        int                int
         |                 |
         v                 v
      integer            integer
         |
         v
        JSON Schema
```

Quando o modelo chama:

```json
{
    "a": "10",
    "b": "20"
}
```

a execução é:

```text
LLM tool call
     |
     v
ToolRegistry
     |
     v
FunctionTool
     |
     +----------------------+
     |                      |
     v                      v
"a" = "10"             "b" = "20"
     |                      |
     v                      v
parse<int>()            parse<int>()
     |                      |
     v                      v
    10                     20
     \                      /
      \                    /
       +--------+---------+
                |
                v
           add(10, 20)
                |
                v
               30
```

---

# 13. Metadados dos argumentos

Podemos definir:

```cpp
struct ToolParam
{
    std::string name;
    std::string description;
    bool required = true;
};
```

Registro:

```cpp
registry.add_function(
    "add",
    "Adds two integers",
    add,
    {
        {"a", "First integer"},
        {"b", "Second integer"}
    }
);
```

O índice do metadado corresponde ao índice do argumento:

```text
ToolParam[0] -> function arg<0>
ToolParam[1] -> function arg<1>
```

Portanto:

```text
"a" -> int
"b" -> int
```

É importante verificar:

```cpp
params.size() == traits::arity
```

durante o registro.

Caso contrário, o sistema deve rejeitar o registro da tool.

---

# 14. Uso de macros apenas como açúcar sintático

Não é necessário usar macros para implementar o sistema.

Templates devem executar a lógica real.

Entretanto, um pequeno macro pode tornar o registro mais agradável:

```cpp
#define TOOL_ARG(name, description) \
    ToolParam{#name, description}
```

Assim:

```cpp
registry.add_function(
    "add",
    "Adds two integers",
    add,
    {
        TOOL_ARG(a, "First integer"),
        TOOL_ARG(b, "Second integer")
    }
);
```

O operador:

```cpp
#name
```

transforma:

```cpp
a
```

em:

```cpp
"a"
```

Isso evita repetir o nome entre aspas.

---

# 15. Lambdas

O mecanismo também pode suportar lambdas.

Exemplo:

```cpp
auto multiply = [](double a, double b)
{
    return a * b;
};
```

Uma lambda em C++ é internamente um objeto com um:

```cpp
operator()
```

Para permitir que `function_traits` descubra sua assinatura, podemos usar:

```cpp
template<typename T>
struct function_traits
    : function_traits<decltype(&T::operator())>
{
};
```

E uma especialização para métodos `const`:

```cpp
template<
    typename C,
    typename R,
    typename... Args
>
struct function_traits<R(C::*)(Args...) const>
{
    using return_type = R;

    static constexpr std::size_t arity =
        sizeof...(Args);

    template<std::size_t I>
    using arg =
        typename std::tuple_element<
            I,
            std::tuple<Args...>
        >::type;
};
```

Agora:

```cpp
auto fn = [](int a, float b)
{
    return a * b;
};
```

permite descobrir:

```text
return = float

arg 0 = int
arg 1 = float
```

---

# 16. `FunctionTool`

Uma arquitetura simples é criar uma implementação de `Tool` que encapsule uma função.

Conceitualmente:

```cpp
template<typename F>
class FunctionTool : public Tool
{
private:

    std::string tool_name;
    std::string tool_description;

    F function;

    std::vector<ToolParam> params;

public:

    FunctionTool(
        std::string name,
        std::string description,
        F fn,
        std::vector<ToolParam> parameters)
        :
        tool_name(std::move(name)),
        tool_description(std::move(description)),
        function(std::move(fn)),
        params(std::move(parameters))
    {
    }

    std::string name() const override
    {
        return tool_name;
    }

    std::string definition() const override
    {
        return build_schema();
    }

    std::string execute(
        const ToolArguments& arguments
    ) override
    {
        return invoke(arguments);
    }
};
```

`FunctionTool` funciona como um adapter:

```text
normal C++ function
        |
        v
   FunctionTool
        |
        v
       Tool
        |
        v
   ToolRegistry
```

Assim o restante da arquitetura não precisa saber se a tool veio de uma classe ou de uma função.

---

# 17. `ToolRegistry::add_function`

O `ToolRegistry` pode receber um novo método:

```cpp
template<typename F>
void add_function(
    const std::string& name,
    const std::string& description,
    F function,
    std::vector<ToolParam> params)
{
    using ToolType = FunctionTool<std::decay_t<F>>;

    auto tool = std::make_shared<ToolType>(
        name,
        description,
        std::move(function),
        std::move(params)
    );

    add(tool);
}
```

Assim:

```cpp
registry.add_function(
    "add",
    "Adds two integers",
    add,
    {
        {"a", "First value"},
        {"b", "Second value"}
    }
);
```

é transformado internamente em:

```text
FunctionTool<int(*)(int,int)>
```

---

# 18. Tipos de retorno

O tipo de retorno também pode ser deduzido.

Exemplo:

```cpp
int add(int a, int b);
```

resulta em:

```text
return_type = int
```

Depois da execução:

```cpp
int result = add(10, 20);
```

podemos converter para o formato retornado ao modelo.

Por exemplo:

```cpp
template<typename T>
std::string tool_result_to_string(T&& value)
{
    using U = std::decay_t<T>;

    if constexpr (std::is_same_v<U, std::string>)
    {
        return value;
    }
    else if constexpr (std::is_same_v<U, bool>)
    {
        return value ? "true" : "false";
    }
    else
    {
        return std::to_string(value);
    }
}
```

Também é útil suportar:

```cpp
void
```

Nesse caso:

```cpp
if constexpr (
    std::is_void_v<typename traits::return_type>
)
{
    invoke_function(...);

    return "ok";
}
else
{
    auto result = invoke_function(...);

    return tool_result_to_string(result);
}
```

---

# 19. Exemplo completo

Função:

```cpp
double calculate_area(
    double width,
    double height)
{
    return width * height;
}
```

Registro:

```cpp
registry.add_function(
    "calculate_area",
    "Calculates the area of a rectangle",
    calculate_area,
    {
        {"width",  "Rectangle width"},
        {"height", "Rectangle height"}
    }
);
```

O compilador deduz:

```text
return_type = double

arg<0> = double
arg<1> = double
```

O framework gera:

```json
{
  "name": "calculate_area",
  "description": "Calculates the area of a rectangle",
  "parameters": {
    "type": "object",
    "properties": {
      "width": {
        "type": "number",
        "description": "Rectangle width"
      },
      "height": {
        "type": "number",
        "description": "Rectangle height"
      }
    },
    "required": [
      "width",
      "height"
    ]
  }
}
```

O modelo pode produzir:

```json
{
    "width": "10.5",
    "height": "3.0"
}
```

O binder executa:

```cpp
calculate_area(
    parse_tool_value<double>("10.5"),
    parse_tool_value<double>("3.0")
);
```

equivalente a:

```cpp
calculate_area(
    10.5,
    3.0
);
```

Resultado:

```text
31.5
```

que é convertido novamente para texto e enviado ao modelo.

---

# 20. Suporte a funções livres e lambdas

Com a infraestrutura anterior podemos suportar:

## Função normal

```cpp
int add(int a, int b)
{
    return a + b;
}
```

```cpp
registry.add_function(
    "add",
    "Adds two integers",
    add,
    {
        {"a", "First number"},
        {"b", "Second number"}
    }
);
```

## Lambda

```cpp
registry.add_function(
    "square",
    "Returns the square of a number",
    [](double value)
    {
        return value * value;
    },
    {
        {"value", "Input number"}
    }
);
```

Os dois casos são transformados para a mesma abstração interna:

```text
FunctionTool<F>
```

---

# 21. Possível API final

Uma API simples seria:

```cpp
ToolRegistry registry;

registry.add_function(
    "add",
    "Adds two integers",
    add,
    {
        TOOL_ARG(a, "First integer"),
        TOOL_ARG(b, "Second integer")
    }
);
```

Também:

```cpp
registry.add_function(
    "temperature",
    "Converts Celsius to Fahrenheit",
    [](double celsius)
    {
        return celsius * 1.8 + 32.0;
    },
    {
        TOOL_ARG(celsius, "Temperature in Celsius")
    }
);
```

Isso é suficientemente explícito para documentação e suficientemente simples para não gerar muito boilerplate.

---

# 22. Manter compatibilidade com tools complexas

Não é necessário remover a interface tradicional baseada em classes.

Ela continua útil para tools com:

- estado interno;
- conexões de rede;
- recursos externos;
- banco de dados;
- caches;
- dependências;
- múltiplas operações internas.

Por exemplo:

```cpp
class DatabaseTool : public Tool
{
private:

    DatabaseConnection connection;

public:

    ...
};
```

Essa tool pode continuar sendo registrada como antes:

```cpp
registry.add<DatabaseTool>();
```

Enquanto tools pequenas usam:

```cpp
registry.add_function(...);
```

Portanto existem dois níveis:

```text
             ToolRegistry
            /            \
           /              \
          v                v

   FunctionTool        Complex Tool
       |                   |
       v                   v

 C++ function          class Tool
```

---

# 23. Organização sugerida

Uma possível estrutura:

```text
include/
    tool.h
    tool_function.h

src/
    tool.cpp

tools/
    math_tools.cpp
    system_tools.cpp
    file_tools.cpp
```

`tool.h` mantém:

```text
Tool
ToolRegistry
ToolParam
```

`tool_function.h` contém:

```text
function_traits
json_type
parse_tool_value
FunctionTool
invoke_function
```

Arquivos específicos registram grupos de funções:

```cpp
void register_math_tools(
    ToolRegistry& registry)
{
    registry.add_function(
        "add",
        "Adds two integers",
        add,
        {
            TOOL_ARG(a, "First value"),
            TOOL_ARG(b, "Second value")
        }
    );
}
```

No `main.cpp`:

```cpp
ToolRegistry registry;

register_system_tools(registry);
register_math_tools(registry);
register_file_tools(registry);
```

---

# 24. Arquitetura geral

```text
                    MiniCPM5
                       |
                       v
                 Tool request
                       |
                       v
                  ToolRegistry
                       |
                       v
                 FunctionTool
                       |
         +-------------+-------------+
         |                           |
         v                           v
   function_traits             Tool metadata
         |                           |
         v                           v
argument C++ types              argument names
         |                           |
         +-------------+-------------+
                       |
                       v
                  JSON Schema
                       |
                       v
                   MiniCPM
                       |
                       v
                 Tool call
                       |
                       v
             parse_tool_value<T>
                       |
                       v
                C++ function
                       |
                       v
                    result
                       |
                       v
            tool_result_to_string
                       |
                       v
                    Model
```

---

# 25. Vantagens

Essa abordagem possui algumas vantagens importantes.

### Menos boilerplate

Uma função simples não precisa de uma classe inteira.

Antes:

```text
Tool subclass
name()
definition()
execute()
argument conversion
JSON schema
```

Depois:

```cpp
int add(int a, int b);
```

mais:

```cpp
registry.add_function(...);
```

### Tipos verificados pelo compilador

A assinatura real da função é usada.

Não existe necessidade de declarar manualmente:

```text
a = integer
b = integer
```

se a função já contém:

```cpp
int add(int a, int b);
```

### Sem reflection externa

A solução funciona em C++17 utilizando:

```text
templates
std::tuple
std::decay_t
std::index_sequence
if constexpr
```

Não depende de reflection moderna do C++ nem de bibliotecas externas.

### Mantém a arquitetura existente

O `FunctionTool` continua implementando a interface:

```cpp
Tool
```

Logo o `ToolRegistry`, o loop do modelo e o parser podem continuar praticamente inalterados.

---

# 26. Limitações

A principal limitação é a ausência de reflexão completa de C++17.

O compilador sabe:

```cpp
int foo(int, double);
```

mas não expõe de forma portável:

```text
primeiro argumento se chama "count"
segundo argumento se chama "threshold"
```

Por isso os nomes precisam ser fornecidos.

Outra limitação é que tipos complexos precisam de conversores específicos.

Por exemplo:

```cpp
std::vector<int>
```

poderia futuramente ser mapeado para:

```json
{
    "type": "array",
    "items": {
        "type": "integer"
    }
}
```

Da mesma forma:

```cpp
std::optional<T>
```

poderia indicar automaticamente que o parâmetro não é obrigatório.

---

# 27. Extensões futuras

A arquitetura permite evoluções interessantes.

## `std::optional`

Exemplo:

```cpp
void search(
    std::string query,
    std::optional<int> limit);
```

O binder poderia detectar automaticamente:

```text
query -> required
limit -> optional
```

## `std::vector`

```cpp
int sum(
    const std::vector<int>& values);
```

poderia produzir:

```json
{
    "type": "array",
    "items": {
        "type": "integer"
    }
}
```

## Enums

```cpp
enum class Unit
{
    Celsius,
    Fahrenheit
};
```

poderia gerar:

```json
{
    "type": "string",
    "enum": [
        "Celsius",
        "Fahrenheit"
    ]
}
```

## Structs

Com um mecanismo adicional de descrição, structs poderiam ser transformadas em objetos JSON.

Exemplo:

```cpp
struct Point
{
    double x;
    double y;
};
```

conceitualmente:

```json
{
    "type": "object",
    "properties": {
        "x": {
            "type": "number"
        },
        "y": {
            "type": "number"
        }
    }
}
```

---

# 28. Conclusão

O ponto principal da solução é separar:

```text
informação que o compilador já possui
```

de:

```text
informação semântica que o usuário precisa fornecer
```

O compilador já conhece:

```text
quantidade de argumentos
tipos dos argumentos
tipo de retorno
```

Portanto não faz sentido exigir essas informações novamente durante o registro da tool.

O desenvolvedor precisa fornecer apenas:

```text
nome da tool
descrição da tool
nome dos parâmetros
descrição dos parâmetros
```

Assim:

```cpp
int add(int a, int b);
```

pode ser registrado como:

```cpp
registry.add_function(
    "add",
    "Adds two integers",
    add,
    {
        TOOL_ARG(a, "First integer"),
        TOOL_ARG(b, "Second integer")
    }
);
```

Internamente:

```text
             int add(int, int)
                     |
                     v
             function_traits
                     |
          +----------+----------+
          |                     |
          v                     v
       arg 0                  arg 1
        int                    int
          |                     |
          v                     v
      integer                integer
          \                     /
           +---------+---------+
                     |
                     v
                 JSON Schema
                     |
                     v
                  MiniCPM
                     |
                     v
                  tool call
                     |
                     v
             FunctionTool
                     |
                     v
               add(10, 20)
                     |
                     v
                    30
```

Essa abordagem mantém a implementação simples, compatível com C++17 e adequada à arquitetura existente do `MiniCPM5.cpp`.