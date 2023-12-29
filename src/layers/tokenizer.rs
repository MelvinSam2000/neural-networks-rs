use nlprule as nlp;

pub struct Tokenizer<const N: usize> {
    token: nlp::Tokenizer,
    rules: nlp::Rules,
}

impl<const N: usize> Tokenizer<N> {
    pub fn new(
        tokenizer_path: String,
        rules_path: String,
    ) -> Self {
        let token =
            nlp::Tokenizer::new(tokenizer_path.as_str())
                .expect("Could not open tokenizer file");
        let rules = nlp::Rules::new(rules_path.as_str())
            .expect("Could not open rules file");
        Self { token, rules }
    }

    pub fn tokenize(
        &self,
        sentence: String,
    ) -> [String; N] {
        let v = self
            .token
            .pipe(&sentence)
            //.map(|token| {
            //    token.
            //})
            .collect::<Vec<_>>();
        todo!()
    }
}

#[test]
//#[ignore]
fn test_tokenizer() {
    let tk =
        nlp::Tokenizer::new("tokenizer/en_tokenizer.bin")
            .unwrap();
    tk.pipe("Hello! I like  2 apples")
        .flat_map(|s| {
            s.tokens().iter().cloned().collect::<Vec<_>>()
        })
        .for_each(|token| {
            println!(
                "{} - {:?}",
                token.word().as_str(),
                token
                    .word()
                    .tags()
                    .get(0)
                    .map(|tag| tag.pos().as_str())
            );
        })
}
